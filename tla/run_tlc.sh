#!/bin/bash
# ============================================================================
# Octopus BFT — TLA+ Model Checking Script
# ============================================================================
# Prerequisites:
#   - Java 11+ installed
#   - TLA+ tools: download tla2tools.jar from
#     https://github.com/tlaplus/tlaplus/releases
#   - Place tla2tools.jar in this directory or set TLA2TOOLS env var
#
# Usage:
#   chmod +x run_tlc.sh
#   ./run_tlc.sh                    # Run all specs
#   ./run_tlc.sh OctopusSafety      # Run specific spec
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TLA2TOOLS="${TLA2TOOLS:-${SCRIPT_DIR}/tla2tools.jar}"
JAVA="${JAVA:-java}"
WORKERS="${TLC_WORKERS:-auto}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_prereqs() {
    if ! command -v "${JAVA}" &> /dev/null; then
        echo -e "${RED}ERROR: Java not found. Install Java 11+ first.${NC}"
        exit 1
    fi
    if [[ ! -f "${TLA2TOOLS}" ]]; then
        echo -e "${RED}ERROR: tla2tools.jar not found at ${TLA2TOOLS}${NC}"
        echo "Download from: https://github.com/tlaplus/tlaplus/releases"
        echo "Or set TLA2TOOLS=/path/to/tla2tools.jar"
        exit 1
    fi
}

run_spec() {
    local spec_name="$1"
    local spec_file="${SCRIPT_DIR}/${spec_name}.tla"
    local cfg_file="${SCRIPT_DIR}/${spec_name}.cfg"

    if [[ ! -f "${spec_file}" ]]; then
        echo -e "${RED}ERROR: ${spec_file} not found${NC}"
        return 1
    fi
    if [[ ! -f "${cfg_file}" ]]; then
        echo -e "${RED}ERROR: ${cfg_file} not found${NC}"
        return 1
    fi

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}  Running TLC on: ${spec_name}${NC}"
    echo -e "${YELLOW}  Config: ${cfg_file}${NC}"
    echo -e "${YELLOW}  Workers: ${WORKERS}${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    local start_time
    start_time=$(date +%s)

    if "${JAVA}" -XX:+UseParallelGC \
        -Xmx4g \
        -cp "${TLA2TOOLS}" tlc2.TLC \
        -workers "${WORKERS}" \
        -config "${cfg_file}" \
        -terse \
        -cleanup \
        "${spec_file}" 2>&1; then
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo -e "${GREEN}✅ ${spec_name}: PASSED (${elapsed}s)${NC}"
        return 0
    else
        local end_time
        end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        echo -e "${RED}❌ ${spec_name}: FAILED (${elapsed}s)${NC}"
        return 1
    fi
}

main() {
    check_prereqs

    local specs=("OctopusSafety" "OctopusMultiLeader" "OctopusReconfiguration" "OctopusComposed")

    if [[ $# -gt 0 ]]; then
        specs=("$@")
    fi

    echo "╔══════════════════════════════════════════════════════╗"
    echo "║  Octopus BFT — TLA+ Model Checking                 ║"
    echo "║  Specs: ${#specs[@]}                                         ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo ""

    local pass=0
    local fail=0

    for spec in "${specs[@]}"; do
        if run_spec "${spec}"; then
            ((pass++))
        else
            ((fail++))
        fi
        echo ""
    done

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "Results: ${GREEN}${pass} passed${NC}, ${RED}${fail} failed${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [[ ${fail} -gt 0 ]]; then
        exit 1
    fi
}

main "$@"
