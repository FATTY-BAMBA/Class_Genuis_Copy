// src/index.ts - Cloudflare Worker Proxy for Class Genius
// Provides: Fixed API endpoint, Rate limiting, API key management, Logging

export interface Env {
  // Secrets (set via wrangler secret put)
  RUNPOD_API_KEY: string;
  RUNPOD_ENDPOINT_ID: string;
  
  // KV Namespaces
  API_KEYS: KVNamespace;      // Client API keys
  RATE_LIMITER: KVNamespace;  // Rate limit counters
  LOGS: KVNamespace;          // Request logs
  JOBS: KVNamespace;          // Job status cache
}

// ==================== Types ====================

interface ClientData {
  clientId: string;
  name: string;
  rateLimit?: number;
  active: boolean;
  createdAt?: string;
  allowedEndpoints?: string[];
}

interface RateLimitInfo {
  count: number;
  resetAt: number;
}

interface LogEntry {
  timestamp: string;
  clientId: string;
  endpoint: string;
  method: string;
  status: number;
  duration: number;
  jobId?: string;
  error?: string;
}

interface VideoInfo {
  Id: string;
  TeamId: string;
  SectionNo: number;
  SectionTitle?: string;
  Units?: Array<{ UnitNo: number; Title: string; Time?: string }>;
  OriginalFilename?: string;
  CreatedAt?: string;
}

interface ProcessRequest {
  video_url?: string;
  play_url?: string;
  PlayUrl?: string;
  video_info?: VideoInfo;
  VideoInfo?: VideoInfo;
  num_questions?: number;
  num_pages?: number;
  webhook_url?: string;
}

// ==================== Config ====================

const RATE_LIMIT_DEFAULT = {
  windowMs: 60 * 1000,  // 1 minute
  maxRequests: 30,      // 30 requests per minute
};

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, X-API-Key, Authorization',
};

// ==================== Utilities ====================

function jsonResponse(data: any, status: number, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...CORS_HEADERS,
      ...headers,
    },
  });
}

async function logRequest(env: Env, entry: LogEntry): Promise<void> {
  try {
    const logKey = `log:${entry.timestamp}:${crypto.randomUUID().slice(0, 8)}`;
    await env.LOGS.put(logKey, JSON.stringify(entry), {
      expirationTtl: 60 * 60 * 24 * 30, // 30 days
    });
  } catch (e) {
    console.error('Failed to log request:', e);
  }
}

// ==================== Middleware ====================

async function validateApiKey(
  request: Request,
  env: Env
): Promise<{ valid: boolean; client?: ClientData; error?: string }> {
  const apiKey = request.headers.get('X-API-Key') || request.headers.get('Authorization')?.replace('Bearer ', '');
  
  if (!apiKey) {
    return { valid: false, error: 'Missing API key. Include X-API-Key header.' };
  }

  try {
    const clientData = await env.API_KEYS.get(apiKey, 'json') as ClientData | null;
    
    if (!clientData) {
      return { valid: false, error: 'Invalid API key.' };
    }
    
    if (!clientData.active) {
      return { valid: false, error: 'API key is inactive. Contact support.' };
    }
    
    return { valid: true, client: clientData };
  } catch (e) {
    console.error('API key validation error:', e);
    return { valid: false, error: 'Authentication error.' };
  }
}

async function checkRateLimit(
  clientId: string,
  maxRequests: number,
  env: Env
): Promise<{ allowed: boolean; remaining: number; resetAt: number }> {
  const rateLimitKey = `rate:${clientId}`;
  const now = Date.now();
  
  try {
    let info = await env.RATE_LIMITER.get(rateLimitKey, 'json') as RateLimitInfo | null;
    
    // Reset if window expired
    if (!info || now > info.resetAt) {
      info = { count: 0, resetAt: now + RATE_LIMIT_DEFAULT.windowMs };
    }
    
    if (info.count >= maxRequests) {
      return {
        allowed: false,
        remaining: 0,
        resetAt: info.resetAt,
      };
    }
    
    // Increment counter
    info.count++;
    await env.RATE_LIMITER.put(rateLimitKey, JSON.stringify(info), {
      expirationTtl: 120,
    });
    
    return {
      allowed: true,
      remaining: maxRequests - info.count,
      resetAt: info.resetAt,
    };
  } catch (e) {
    console.error('Rate limit check error:', e);
    // Allow on error (fail open)
    return { allowed: true, remaining: maxRequests, resetAt: now + RATE_LIMIT_DEFAULT.windowMs };
  }
}

// ==================== Route Handlers ====================

async function handleProcess(
  request: Request,
  env: Env,
  client: ClientData
): Promise<Response> {
  let body: ProcessRequest;
  
  try {
    body = await request.json() as ProcessRequest;
  } catch (e) {
    return jsonResponse({ error: 'Invalid JSON body' }, 400);
  }

  // Normalize video URL
  const videoUrl = body.video_url || body.play_url || body.PlayUrl;
  if (!videoUrl) {
    return jsonResponse({ error: 'Missing required field: video_url' }, 400);
  }

  // Normalize video info
  const videoInfo = body.video_info || body.VideoInfo;
  if (!videoInfo) {
    return jsonResponse({ error: 'Missing required field: video_info' }, 400);
  }

  // Validate required video_info fields
  if (!videoInfo.Id || !videoInfo.TeamId || videoInfo.SectionNo === undefined) {
    return jsonResponse({ 
      error: 'video_info must contain: Id, TeamId, SectionNo' 
    }, 400);
  }

  // Build RunPod request
  const runpodPayload = {
    input: {
      video_url: videoUrl,
      video_info: videoInfo,
      num_questions: body.num_questions || 10,
      num_pages: body.num_pages || 3,
      webhook_url: body.webhook_url,
    },
  };

  try {
    // Submit to RunPod (async mode)
    const runpodResponse = await fetch(
      `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/run`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${env.RUNPOD_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(runpodPayload),
      }
    );

    const result = await runpodResponse.json() as { id?: string; status?: string };
    
    if (!runpodResponse.ok) {
      console.error('RunPod error:', result);
      return jsonResponse({
        error: 'Processing service error',
        details: result,
      }, runpodResponse.status);
    }

    // Cache job info
    if (result.id) {
      await env.JOBS.put(`job:${result.id}`, JSON.stringify({
        clientId: client.clientId,
        videoId: videoInfo.Id,
        createdAt: new Date().toISOString(),
      }), { expirationTtl: 60 * 60 * 24 }); // 24 hours
    }

    return jsonResponse({
      success: true,
      job_id: result.id,
      status: result.status || 'IN_QUEUE',
      message: 'Video processing started. Use /api/status?job_id=XXX to check progress.',
    }, 202);

  } catch (e) {
    console.error('RunPod request failed:', e);
    return jsonResponse({
      error: 'Failed to submit processing job',
      details: (e as Error).message,
    }, 500);
  }
}

async function handleProcessSync(
  request: Request,
  env: Env,
  client: ClientData
): Promise<Response> {
  // Similar to handleProcess but uses /runsync endpoint
  // WARNING: This will timeout for long videos (>30s processing)
  
  let body: ProcessRequest;
  try {
    body = await request.json() as ProcessRequest;
  } catch (e) {
    return jsonResponse({ error: 'Invalid JSON body' }, 400);
  }

  const videoUrl = body.video_url || body.play_url || body.PlayUrl;
  const videoInfo = body.video_info || body.VideoInfo;

  if (!videoUrl || !videoInfo) {
    return jsonResponse({ error: 'Missing video_url or video_info' }, 400);
  }

  const runpodPayload = {
    input: {
      video_url: videoUrl,
      video_info: videoInfo,
      num_questions: body.num_questions || 10,
      num_pages: body.num_pages || 3,
    },
  };

  try {
    const runpodResponse = await fetch(
      `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/runsync`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${env.RUNPOD_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(runpodPayload),
      }
    );

    const result = await runpodResponse.json();
    return jsonResponse(result, runpodResponse.status);

  } catch (e) {
    console.error('RunPod sync request failed:', e);
    return jsonResponse({
      error: 'Processing failed',
      details: (e as Error).message,
    }, 500);
  }
}

async function handleStatus(
  request: Request,
  env: Env,
  client: ClientData
): Promise<Response> {
  const url = new URL(request.url);
  const jobId = url.searchParams.get('job_id');

  if (!jobId) {
    return jsonResponse({ error: 'Missing required parameter: job_id' }, 400);
  }

  // Verify job belongs to client (optional security check)
  const jobInfo = await env.JOBS.get(`job:${jobId}`, 'json') as { clientId?: string } | null;
  if (jobInfo && jobInfo.clientId !== client.clientId) {
    return jsonResponse({ error: 'Job not found' }, 404);
  }

  try {
    const runpodResponse = await fetch(
      `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/status/${jobId}`,
      {
        headers: {
          'Authorization': `Bearer ${env.RUNPOD_API_KEY}`,
        },
      }
    );

    const result = await runpodResponse.json();
    return jsonResponse(result, runpodResponse.status);

  } catch (e) {
    console.error('Status check failed:', e);
    return jsonResponse({
      error: 'Failed to check job status',
      details: (e as Error).message,
    }, 500);
  }
}

async function handleCancel(
  request: Request,
  env: Env,
  client: ClientData
): Promise<Response> {
  const url = new URL(request.url);
  const jobId = url.searchParams.get('job_id');

  if (!jobId) {
    return jsonResponse({ error: 'Missing required parameter: job_id' }, 400);
  }

  try {
    const runpodResponse = await fetch(
      `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/cancel/${jobId}`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${env.RUNPOD_API_KEY}`,
        },
      }
    );

    const result = await runpodResponse.json();
    return jsonResponse(result, runpodResponse.status);

  } catch (e) {
    console.error('Cancel request failed:', e);
    return jsonResponse({
      error: 'Failed to cancel job',
      details: (e as Error).message,
    }, 500);
  }
}

async function handleHealth(env: Env): Promise<Response> {
  // Check RunPod endpoint health
  let runpodHealthy = false;
  
  try {
    const response = await fetch(
      `https://api.runpod.ai/v2/${env.RUNPOD_ENDPOINT_ID}/health`,
      {
        headers: {
          'Authorization': `Bearer ${env.RUNPOD_API_KEY}`,
        },
      }
    );
    runpodHealthy = response.ok;
  } catch (e) {
    console.error('Health check failed:', e);
  }

  return jsonResponse({
    status: runpodHealthy ? 'healthy' : 'degraded',
    timestamp: new Date().toISOString(),
    services: {
      proxy: 'healthy',
      runpod: runpodHealthy ? 'healthy' : 'unhealthy',
    },
  }, runpodHealthy ? 200 : 503);
}

// ==================== Main Handler ====================

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const startTime = Date.now();
    const url = new URL(request.url);
    const path = url.pathname;

    // Handle CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: CORS_HEADERS });
    }

    // Public endpoints (no auth required)
    if (path === '/health' || path === '/api/health') {
      return handleHealth(env);
    }

    // API Documentation
    if (path === '/' || path === '/api') {
      return jsonResponse({
        name: 'Class Genius Video Processing API',
        version: '1.0.0',
        endpoints: {
          'POST /api/process': 'Submit video for processing (async)',
          'POST /api/process/sync': 'Submit video for processing (sync, short videos only)',
          'GET /api/status?job_id=XXX': 'Check job status',
          'POST /api/cancel?job_id=XXX': 'Cancel a job',
          'GET /api/health': 'Health check',
        },
        authentication: 'Include X-API-Key header with all requests',
      }, 200);
    }

    // ==================== Authenticated Endpoints ====================
    
    // Validate API key
    const authResult = await validateApiKey(request, env);
    if (!authResult.valid || !authResult.client) {
      return jsonResponse({ error: authResult.error }, 401);
    }
    
    const client = authResult.client;

    // Check rate limit
    const maxRequests = client.rateLimit || RATE_LIMIT_DEFAULT.maxRequests;
    const rateLimit = await checkRateLimit(client.clientId, maxRequests, env);
    
    if (!rateLimit.allowed) {
      const retryAfter = Math.ceil((rateLimit.resetAt - Date.now()) / 1000);
      return jsonResponse(
        { 
          error: 'Rate limit exceeded',
          retryAfter,
          message: `Please wait ${retryAfter} seconds before retrying.`
        },
        429,
        { 'Retry-After': retryAfter.toString() }
      );
    }

    // Route to handler
    let response: Response;
    let jobId: string | undefined;

    try {
      if (path === '/api/process' && request.method === 'POST') {
        response = await handleProcess(request, env, client);
        // Extract job ID from response for logging
        try {
          const body = await response.clone().json() as { job_id?: string };
          jobId = body.job_id;
        } catch {}
        
      } else if (path === '/api/process/sync' && request.method === 'POST') {
        response = await handleProcessSync(request, env, client);
        
      } else if (path === '/api/status' && request.method === 'GET') {
        response = await handleStatus(request, env, client);
        
      } else if (path === '/api/cancel' && request.method === 'POST') {
        response = await handleCancel(request, env, client);
        
      } else {
        response = jsonResponse({ error: 'Not found' }, 404);
      }
    } catch (e) {
      console.error('Handler error:', e);
      response = jsonResponse({
        error: 'Internal server error',
        message: (e as Error).message,
      }, 500);
    }

    // Add rate limit headers
    response.headers.set('X-RateLimit-Limit', maxRequests.toString());
    response.headers.set('X-RateLimit-Remaining', rateLimit.remaining.toString());
    response.headers.set('X-RateLimit-Reset', rateLimit.resetAt.toString());

    // Log request (async, don't wait)
    ctx.waitUntil(
      logRequest(env, {
        timestamp: new Date().toISOString(),
        clientId: client.clientId,
        endpoint: path,
        method: request.method,
        status: response.status,
        duration: Date.now() - startTime,
        jobId,
      })
    );

    return response;
  },
};
