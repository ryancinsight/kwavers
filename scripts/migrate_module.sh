#!/bin/bash
# Script: migrate_module.sh
# Purpose: Automated module migration for deep vertical hierarchy refactoring
# Usage: ./scripts/migrate_module.sh <source> <destination>
# Example: ./scripts/migrate_module.sh domain/core/error core/error

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
SOURCE=$1
DEST=$2

if [ -z "$SOURCE" ] || [ -z "$DEST" ]; then
    echo -e "${RED}❌ Error: Missing arguments${NC}"
    echo "Usage: $0 <source> <destination>"
    echo "Example: $0 domain/core/error core/error"
    exit 1
fi

echo -e "${BLUE}🚚 Module Migration Tool${NC}"
echo -e "${BLUE}========================${NC}"
echo -e "Source:      ${YELLOW}$SOURCE${NC}"
echo -e "Destination: ${YELLOW}$DEST${NC}"
echo ""

# Convert paths to filesystem paths
SRC_PATH="src/${SOURCE}"
DEST_PATH="src/${DEST}"

# Verify source exists
if [ ! -e "$SRC_PATH" ]; then
    echo -e "${RED}❌ Source does not exist: $SRC_PATH${NC}"
    exit 1
fi

# Check if destination already exists
if [ -e "$DEST_PATH" ]; then
    echo -e "${YELLOW}⚠️  Warning: Destination already exists: $DEST_PATH${NC}"
    echo -e "Do you want to merge/overwrite? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted${NC}"
        exit 1
    fi
fi

# Check if working directory is clean
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}⚠️  Warning: You have uncommitted changes${NC}"
    echo -e "It's recommended to commit or stash changes before migration."
    echo -e "Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo -e "${RED}Aborted${NC}"
        exit 1
    fi
fi

# Create destination directory structure
echo -e "${BLUE}📁 Creating destination directory...${NC}"
DEST_DIR=$(dirname "$DEST_PATH")
mkdir -p "$DEST_DIR"

# Count files to be migrated
if [ -d "$SRC_PATH" ]; then
    FILE_COUNT=$(find "$SRC_PATH" -type f -name "*.rs" | wc -l)
    echo -e "${GREEN}Found $FILE_COUNT Rust files to migrate${NC}"
elif [ -f "$SRC_PATH" ]; then
    FILE_COUNT=1
    echo -e "${GREEN}Migrating single file${NC}"
else
    echo -e "${RED}❌ Source is neither file nor directory${NC}"
    exit 1
fi

# Copy files to destination
echo -e "${BLUE}🔄 Copying files...${NC}"
if [ -d "$SRC_PATH" ]; then
    # Copy directory contents
    cp -r "$SRC_PATH"/* "$DEST_PATH"/ 2>/dev/null || {
        # If dest doesn't exist, copy the whole directory
        mkdir -p "$DEST_PATH"
        cp -r "$SRC_PATH"/* "$DEST_PATH"/
    }
else
    # Copy single file
    cp "$SRC_PATH" "$DEST_PATH"
fi
echo -e "${GREEN}✓ Files copied${NC}"

# Update internal imports within moved files
echo -e "${BLUE}🔧 Updating internal imports in moved files...${NC}"
SOURCE_MODULE=$(echo "$SOURCE" | tr '/' ':')
DEST_MODULE=$(echo "$DEST" | tr '/' ':')

if [ -d "$DEST_PATH" ]; then
    # Update imports within the moved directory
    find "$DEST_PATH" -type f -name "*.rs" -exec sed -i "s|crate::${SOURCE_MODULE}|crate::${DEST_MODULE}|g" {} +
    # Also update any domain::core references to core if moving out of domain
    if [[ "$SOURCE" == domain/* ]] && [[ "$DEST" != domain/* ]]; then
        find "$DEST_PATH" -type f -name "*.rs" -exec sed -i "s|crate::domain::core::|crate::core::|g" {} +
    fi
fi
echo -e "${GREEN}✓ Internal imports updated${NC}"

# Update all imports across codebase using Python script
echo -e "${BLUE}🔍 Updating imports across entire codebase...${NC}"
if [ -f "scripts/update_imports.py" ]; then
    python3 scripts/update_imports.py "$SOURCE" "$DEST"
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Import update script failed${NC}"
        echo -e "${YELLOW}⚠️  Manual import updates may be required${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  Warning: update_imports.py not found${NC}"
    echo -e "${YELLOW}   Manual import updates required${NC}"
fi

# Test compilation
echo ""
echo -e "${BLUE}🧪 Testing compilation...${NC}"
if cargo check --all-features 2>&1 | tee migration_check.log; then
    echo -e "${GREEN}✅ Compilation successful${NC}"
    rm -f migration_check.log
else
    echo -e "${RED}❌ Compilation failed${NC}"
    echo -e "${YELLOW}Check migration_check.log for details${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  ROLLBACK OPTIONS:${NC}"
    echo "1. Manual fix: Review errors and fix imports"
    echo "2. Revert: rm -rf $DEST_PATH && git checkout $SRC_PATH"
    exit 1
fi

# Run quick test suite
echo ""
echo -e "${BLUE}🧪 Running quick test suite...${NC}"
echo -e "${YELLOW}(Library tests only for speed)${NC}"
if cargo test --lib --all-features 2>&1 | tee migration_test.log | tail -20; then
    echo -e "${GREEN}✅ Tests passed${NC}"
    rm -f migration_test.log
else
    echo -e "${RED}❌ Tests failed${NC}"
    echo -e "${YELLOW}Check migration_test.log for details${NC}"
    echo ""
    echo -e "${YELLOW}⚠️  Migration may be incomplete${NC}"
    exit 1
fi

# Summary
echo ""
echo -e "${BLUE}📊 Migration Summary${NC}"
echo -e "${BLUE}====================${NC}"
echo -e "${GREEN}✓ Files migrated:      $FILE_COUNT${NC}"
echo -e "${GREEN}✓ Compilation:         PASSED${NC}"
echo -e "${GREEN}✓ Tests:               PASSED${NC}"
echo ""
echo -e "${YELLOW}⚠️  NEXT STEPS:${NC}"
echo "1. Review changes:     git diff"
echo "2. Run full tests:     cargo test --all-features"
echo "3. Check for old refs: grep -r '$SOURCE_MODULE' src/"
echo "4. If successful:      rm -rf $SRC_PATH"
echo "5. Update mod.rs:      Remove/update module declaration"
echo "6. Commit:             git commit -am 'refactor: migrate $SOURCE to $DEST'"
echo ""
echo -e "${GREEN}✅ Migration script complete${NC}"
echo ""
echo -e "${YELLOW}NOTE: Source files at $SRC_PATH are NOT deleted automatically.${NC}"
echo -e "${YELLOW}      Delete manually after verifying everything works.${NC}"

exit 0
