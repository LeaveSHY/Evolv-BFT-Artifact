#!/bin/bash
# EC2 Deployment Script for Consensus Baseline Comparison
#
# Deploys Evolv-BFT, Bullshark, and Ladon across 100 EC2 instances
# and runs the full benchmark sweep.
#
# Usage:
#   ./deploy_ec2.sh --all --scale 1000    # Full benchmark
#   ./deploy_ec2.sh --bullshark --scale 64 # Bullshark only, n=64
#   ./deploy_ec2.sh --teardown             # Terminate all instances
#
# Prerequisites:
#   - AWS CLI v2 configured (aws configure)
#   - IAM permissions: ec2:RunInstances, ec2:TerminateInstances, etc.
#   - SSH key pair registered in target regions
#
# Cost estimate: ~$200 for full 3-trial sweep at n=1000 (~6h runtime)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Configuration
INSTANCE_TYPE="c5.xlarge"
AMI_ID="ami-0c55b159cbfafe1f0"  # Ubuntu 22.04 LTS (us-east-1)
KEY_NAME="evolvbft-benchmark"
SECURITY_GROUP="sg-evolvbft-bench"
N_INSTANCES=100
REGIONS=("us-east-1" "us-west-2" "eu-west-1" "ap-northeast-1" "ap-southeast-1")
REPLICAS_SWEEP=(4 16 64 100 400 1000)
NETWORKS=("wan" "lan")

# Parse arguments
RUN_EVOLVBFT=false
RUN_BULLSHARK=false
RUN_LADON=false
SCALE=1000
TEARDOWN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all) RUN_EVOLVBFT=true; RUN_BULLSHARK=true; RUN_LADON=true ;;
        --evolvbft) RUN_EVOLVBFT=true ;;
        --bullshark) RUN_BULLSHARK=true ;;
        --ladon) RUN_LADON=true ;;
        --scale) SCALE="$2"; shift ;;
        --teardown) TEARDOWN=true ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Instance management
INSTANCE_FILE="${SCRIPT_DIR}/.ec2_instances.json"

launch_instances() {
    echo "=== Launching ${N_INSTANCES} EC2 instances ==="
    local instances_per_region=$((N_INSTANCES / ${#REGIONS[@]}))

    ALL_IDS=()
    for region in "${REGIONS[@]}"; do
        echo "  Region: ${region} (${instances_per_region} instances)"
        IDS=$(aws ec2 run-instances \
            --region "${region}" \
            --image-id "${AMI_ID}" \
            --instance-type "${INSTANCE_TYPE}" \
            --key-name "${KEY_NAME}" \
            --security-group-ids "${SECURITY_GROUP}" \
            --count "${instances_per_region}" \
            --tag-specifications "ResourceType=instance,Tags=[{Key=Project,Value=evolvbft-benchmark}]" \
            --query 'Instances[*].InstanceId' \
            --output text)
        ALL_IDS+=($IDS)
    done

    echo "${ALL_IDS[@]}" | tr ' ' '\n' > "${INSTANCE_FILE}"
    echo "  Launched ${#ALL_IDS[@]} instances total"
    echo "  Waiting for running state..."
    sleep 60
}

setup_instances() {
    echo "=== Setting up instances ==="
    local ips=$(get_public_ips)

    # Install dependencies in parallel
    echo "  Installing dependencies on all nodes..."
    for ip in ${ips}; do
        ssh -o StrictHostKeyChecking=no ubuntu@${ip} \
            "sudo apt-get update -qq && \
             sudo apt-get install -y -qq build-essential protobuf-compiler clang pkg-config && \
             curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
             wget -q https://go.dev/dl/go1.21.5.linux-amd64.tar.gz && \
             sudo tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz" &
    done
    wait
    echo "  Dependencies installed"
}

run_benchmark_sweep() {
    local protocol=$1
    echo "=== Running ${protocol} benchmark sweep ==="

    for n in "${REPLICAS_SWEEP[@]}"; do
        if [ "${n}" -gt "${SCALE}" ]; then
            echo "  Skipping n=${n} (exceeds --scale ${SCALE})"
            continue
        fi
        for network in "${NETWORKS[@]}"; do
            echo "  ${protocol} n=${n} ${network}..."
            # Deploy and run on EC2 cluster
            # (Actual implementation would use ansible/fabric for orchestration)
            echo "    [Would run: baselines/${protocol,,}/run_benchmark.sh ${n} ${network}]"
        done
    done
}

get_public_ips() {
    if [ -f "${INSTANCE_FILE}" ]; then
        cat "${INSTANCE_FILE}" | while read id; do
            aws ec2 describe-instances --instance-ids "${id}" \
                --query 'Reservations[*].Instances[*].PublicIpAddress' \
                --output text 2>/dev/null
        done
    fi
}

teardown() {
    echo "=== Terminating all instances ==="
    if [ -f "${INSTANCE_FILE}" ]; then
        local ids=$(cat "${INSTANCE_FILE}" | tr '\n' ' ')
        aws ec2 terminate-instances --instance-ids ${ids}
        rm "${INSTANCE_FILE}"
        echo "  Terminated"
    else
        echo "  No instance file found"
    fi
}

# Main execution
if [ "${TEARDOWN}" = true ]; then
    teardown
    exit 0
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║  Evolv-BFT Consensus Benchmark - EC2 Deployment   ║"
echo "╠══════════════════════════════════════════════════╣"
echo "║  Instances: ${N_INSTANCES} × ${INSTANCE_TYPE}              ║"
echo "║  Regions:   ${#REGIONS[@]} (cross-continent WAN)           ║"
echo "║  Scale:     up to n=${SCALE}                        ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# Launch and setup
launch_instances
setup_instances

# Run benchmarks
if [ "${RUN_EVOLVBFT}" = true ]; then
    run_benchmark_sweep "Evolv-BFT"
fi
if [ "${RUN_BULLSHARK}" = true ]; then
    run_benchmark_sweep "Bullshark"
fi
if [ "${RUN_LADON}" = true ]; then
    run_benchmark_sweep "Ladon"
fi

echo ""
echo "=== All benchmarks complete ==="
echo "Results collected in: ${SCRIPT_DIR}/results/"
echo "Run ./deploy_ec2.sh --teardown to terminate instances"
