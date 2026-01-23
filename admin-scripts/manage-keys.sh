#!/bin/bash
# admin-scripts/manage-keys.sh
# Helper scripts for managing Class Genius API keys

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==================== Functions ====================

show_help() {
    echo "Class Genius API Key Management"
    echo ""
    echo "Usage: ./manage-keys.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  list                    List all API keys"
    echo "  add <key> <client_id> <name> [rate_limit]"
    echo "                          Add a new API key"
    echo "  get <key>               Get details for an API key"
    echo "  deactivate <key>        Deactivate an API key"
    echo "  activate <key>          Reactivate an API key"
    echo "  delete <key>            Delete an API key"
    echo "  generate                Generate a new random API key"
    echo ""
    echo "Examples:"
    echo "  ./manage-keys.sh list"
    echo "  ./manage-keys.sh add my-api-key client-001 'Main Client' 60"
    echo "  ./manage-keys.sh deactivate my-api-key"
    echo ""
}

generate_key() {
    # Generate a secure random API key
    openssl rand -base64 32 | tr -d '/+=' | cut -c1-32
}

list_keys() {
    echo -e "${GREEN}Listing all API keys...${NC}"
    wrangler kv:key list --binding=API_KEYS | jq -r '.[].name'
}

add_key() {
    local key=$1
    local client_id=$2
    local name=$3
    local rate_limit=${4:-30}
    
    if [ -z "$key" ] || [ -z "$client_id" ] || [ -z "$name" ]; then
        echo -e "${RED}Error: Missing required arguments${NC}"
        echo "Usage: ./manage-keys.sh add <key> <client_id> <name> [rate_limit]"
        exit 1
    fi
    
    local created_at=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local data=$(cat <<EOF
{
  "clientId": "$client_id",
  "name": "$name",
  "rateLimit": $rate_limit,
  "active": true,
  "createdAt": "$created_at"
}
EOF
)
    
    echo -e "${GREEN}Adding API key...${NC}"
    echo "Key: $key"
    echo "Client ID: $client_id"
    echo "Name: $name"
    echo "Rate Limit: $rate_limit requests/minute"
    
    wrangler kv:key put --binding=API_KEYS "$key" "$data"
    
    echo -e "${GREEN}✅ API key added successfully${NC}"
}

get_key() {
    local key=$1
    
    if [ -z "$key" ]; then
        echo -e "${RED}Error: Missing key argument${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Getting API key details...${NC}"
    wrangler kv:key get --binding=API_KEYS "$key" | jq .
}

update_key_status() {
    local key=$1
    local active=$2
    
    if [ -z "$key" ]; then
        echo -e "${RED}Error: Missing key argument${NC}"
        exit 1
    fi
    
    # Get existing data
    local existing=$(wrangler kv:key get --binding=API_KEYS "$key" 2>/dev/null)
    
    if [ -z "$existing" ]; then
        echo -e "${RED}Error: API key not found${NC}"
        exit 1
    fi
    
    # Update active status
    local updated=$(echo "$existing" | jq ".active = $active")
    
    wrangler kv:key put --binding=API_KEYS "$key" "$updated"
    
    if [ "$active" = "true" ]; then
        echo -e "${GREEN}✅ API key activated${NC}"
    else
        echo -e "${YELLOW}⚠️ API key deactivated${NC}"
    fi
}

delete_key() {
    local key=$1
    
    if [ -z "$key" ]; then
        echo -e "${RED}Error: Missing key argument${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Are you sure you want to delete this API key? (y/N)${NC}"
    read -r confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        wrangler kv:key delete --binding=API_KEYS "$key"
        echo -e "${GREEN}✅ API key deleted${NC}"
    else
        echo "Cancelled"
    fi
}

# ==================== Main ====================

case "${1:-help}" in
    list)
        list_keys
        ;;
    add)
        add_key "$2" "$3" "$4" "$5"
        ;;
    get)
        get_key "$2"
        ;;
    deactivate)
        update_key_status "$2" "false"
        ;;
    activate)
        update_key_status "$2" "true"
        ;;
    delete)
        delete_key "$2"
        ;;
    generate)
        echo -e "${GREEN}Generated API key:${NC}"
        generate_key
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        show_help
        exit 1
        ;;
esac
