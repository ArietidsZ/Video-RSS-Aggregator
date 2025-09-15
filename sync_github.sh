#!/bin/bash
# GitHub Bidirectional Sync Script
# For continuous collaboration between developers

echo "🔄 GitHub Sync Tool for Video RSS Aggregator"
echo "==========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if there are local changes
check_local_changes() {
    if [[ -n $(git status -s) ]]; then
        return 0  # Has changes
    else
        return 1  # No changes
    fi
}

# Function to sync with GitHub
sync_github() {
    echo -e "${YELLOW}📡 Fetching latest from GitHub...${NC}"
    git fetch origin main

    # Check if we're behind
    LOCAL=$(git rev-parse @)
    REMOTE=$(git rev-parse @{u})
    BASE=$(git merge-base @ @{u})

    if [ $LOCAL = $REMOTE ]; then
        echo -e "${GREEN}✅ Already up to date with GitHub${NC}"
    elif [ $LOCAL = $BASE ]; then
        echo -e "${YELLOW}⬇️  Pulling changes from GitHub...${NC}"
        git pull origin main
        echo -e "${GREEN}✅ Updated from GitHub${NC}"
    elif [ $REMOTE = $BASE ]; then
        echo -e "${YELLOW}⬆️  You have local changes to push${NC}"
        read -p "Push changes to GitHub? (y/n): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git push origin main
            echo -e "${GREEN}✅ Pushed to GitHub${NC}"
        fi
    else
        echo -e "${RED}⚠️  Diverged from GitHub - manual merge needed${NC}"
        echo "Run: git pull --rebase origin main"
    fi
}

# Function for continuous sync mode
continuous_sync() {
    echo -e "${GREEN}🔄 Starting continuous sync mode...${NC}"
    echo "Will check for changes every 30 seconds"
    echo "Press Ctrl+C to stop"

    while true; do
        sync_github
        echo -e "${YELLOW}💤 Waiting 30 seconds...${NC}"
        sleep 30
    done
}

# Main menu
echo ""
echo "Select sync option:"
echo "1) Sync once"
echo "2) Continuous sync (every 30 seconds)"
echo "3) Commit and sync"
echo "4) Setup auto-sync (git hooks)"

read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        sync_github
        ;;
    2)
        continuous_sync
        ;;
    3)
        if check_local_changes; then
            echo -e "${YELLOW}📝 Local changes detected${NC}"
            git status -s
            read -p "Enter commit message: " msg
            git add -A
            git commit -m "$msg"
            git push origin main
            echo -e "${GREEN}✅ Changes committed and pushed${NC}"
        else
            echo -e "${GREEN}No local changes to commit${NC}"
            sync_github
        fi
        ;;
    4)
        echo -e "${YELLOW}⚙️  Setting up auto-sync with git hooks...${NC}"

        # Create post-commit hook
        cat > .git/hooks/post-commit << 'EOF'
#!/bin/bash
# Auto-push after commit
echo "🔄 Auto-pushing to GitHub..."
git push origin main
EOF
        chmod +x .git/hooks/post-commit

        # Create post-merge hook
        cat > .git/hooks/post-merge << 'EOF'
#!/bin/bash
# Notify after pull
echo "✅ Synced with GitHub successfully"
EOF
        chmod +x .git/hooks/post-merge

        echo -e "${GREEN}✅ Auto-sync hooks installed${NC}"
        echo "Your commits will now auto-push to GitHub"
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✨ Sync complete!${NC}"