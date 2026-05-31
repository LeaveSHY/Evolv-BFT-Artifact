#!/bin/bash
# deploy_ec2.sh — Provision and configure EC2 instances for Evolv-BFT benchmark.
# Prerequisites: aws CLI configured, ssh key named "evolvbft-bench" in target regions.
#
# Usage:
#   ./deploy_ec2.sh setup    # Create instances in 4 regions
#   ./deploy_ec2.sh teardown # Terminate all instances
#   ./deploy_ec2.sh status   # Show running instances

set -euo pipefail

# --- Configuration ---
KEY_NAME="evolvbft-bench"
INSTANCE_TYPE="c5.4xlarge"   # 16 vCPU, 32 GB RAM
AMI_FAMILY="amzn2"           # Amazon Linux 2 (latest AMI per region)
SECURITY_GROUP="evolvbft-bench-sg"
TAG_KEY="Project"
TAG_VALUE="evolvbft-bench"
NODES_PER_REGION=25
REGIONS=("us-east-1" "us-west-2" "eu-west-1" "ap-southeast-1")

# AMI IDs (Amazon Linux 2, x86_64, updated periodically)
declare -A AMIS=(
  ["us-east-1"]="ami-0c02fb55956c7d316"
  ["us-west-2"]="ami-0ceecbb0f30a9a19a"
  ["eu-west-1"]="ami-0d71ea30463e0ff8d"
  ["ap-southeast-1"]="ami-04d9e855d716227c6"
)

INSTANCE_FILE="ec2_instances.json"

# --- Functions ---

get_latest_ami() {
  local region=$1
  aws ec2 describe-images --region "$region" \
    --owners amazon \
    --filters "Name=name,Values=amzn2-ami-hvm-*-x86_64-gp2" \
              "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text 2>/dev/null || echo "${AMIS[$region]}"
}

ensure_security_group() {
  local region=$1
  local sgid
  sgid=$(aws ec2 describe-security-groups --region "$region" \
    --filters "Name=group-name,Values=$SECURITY_GROUP" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)

  if [ "$sgid" == "None" ] || [ -z "$sgid" ]; then
    sgid=$(aws ec2 create-security-group --region "$region" \
      --group-name "$SECURITY_GROUP" \
      --description "Evolv-BFT benchmark - allow all internal + SSH" \
      --output text --query 'GroupId')
    # Allow SSH
    aws ec2 authorize-security-group-ingress --region "$region" \
      --group-id "$sgid" --protocol tcp --port 22 --cidr 0.0.0.0/0
    # Allow Evolv-BFT ports (8080-8180 for libp2p, 9000-9100 for HTTP API)
    aws ec2 authorize-security-group-ingress --region "$region" \
      --group-id "$sgid" --protocol tcp --port 8080-8180 --cidr 0.0.0.0/0
    aws ec2 authorize-security-group-ingress --region "$region" \
      --group-id "$sgid" --protocol tcp --port 9000-9100 --cidr 0.0.0.0/0
    echo "Created security group $sgid in $region"
  fi
  echo "$sgid"
}

setup() {
  echo "=== Setting up EC2 instances across ${#REGIONS[@]} regions ==="
  local instances=()

  for region in "${REGIONS[@]}"; do
    echo "--- Region: $region ---"
    local ami
    ami=$(get_latest_ami "$region")
    echo "  AMI: $ami"

    local sgid
    sgid=$(ensure_security_group "$region")
    echo "  Security Group: $sgid"

    # Launch 1 instance per region (each will run NODES_PER_REGION processes)
    local result
    result=$(aws ec2 run-instances --region "$region" \
      --image-id "$ami" \
      --instance-type "$INSTANCE_TYPE" \
      --key-name "$KEY_NAME" \
      --security-group-ids "$sgid" \
      --count 1 \
      --tag-specifications "ResourceType=instance,Tags=[{Key=$TAG_KEY,Value=$TAG_VALUE},{Key=Region,Value=$region}]" \
      --output json)

    local instance_id
    instance_id=$(echo "$result" | python3 -c "import sys,json; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])")
    echo "  Launched: $instance_id"
    instances+=("{\"region\":\"$region\",\"instance_id\":\"$instance_id\"}")
  done

  # Wait for all instances to be running
  echo ""
  echo "Waiting for instances to reach 'running' state..."
  for region in "${REGIONS[@]}"; do
    aws ec2 wait instance-running --region "$region" \
      --filters "Name=tag:$TAG_KEY,Values=$TAG_VALUE"
  done

  # Collect public IPs
  echo ""
  echo "=== Instance Details ==="
  local details="["
  local first=true
  for region in "${REGIONS[@]}"; do
    local info
    info=$(aws ec2 describe-instances --region "$region" \
      --filters "Name=tag:$TAG_KEY,Values=$TAG_VALUE" "Name=instance-state-name,Values=running" \
      --query 'Reservations[].Instances[].[InstanceId,PublicIpAddress,PrivateIpAddress]' \
      --output json)

    local id ip priv_ip
    id=$(echo "$info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0][0]) if d else print('')")
    ip=$(echo "$info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0][1]) if d else print('')")
    priv_ip=$(echo "$info" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[0][2]) if d else print('')")

    echo "  $region: $id | public=$ip | private=$priv_ip"

    if [ "$first" = true ]; then first=false; else details+=","; fi
    details+="{\"region\":\"$region\",\"instance_id\":\"$id\",\"public_ip\":\"$ip\",\"private_ip\":\"$priv_ip\"}"
  done
  details+="]"

  echo "$details" | python3 -m json.tool > "$INSTANCE_FILE"
  echo ""
  echo "Instance details saved to $INSTANCE_FILE"
  echo "Next: run './setup_nodes.sh' to install Go and deploy Evolv-BFT binary"
}

teardown() {
  echo "=== Terminating all benchmark instances ==="
  for region in "${REGIONS[@]}"; do
    local ids
    ids=$(aws ec2 describe-instances --region "$region" \
      --filters "Name=tag:$TAG_KEY,Values=$TAG_VALUE" "Name=instance-state-name,Values=running" \
      --query 'Reservations[].Instances[].InstanceId' --output text)
    if [ -n "$ids" ]; then
      echo "  Terminating in $region: $ids"
      aws ec2 terminate-instances --region "$region" --instance-ids $ids > /dev/null
    fi
  done
  echo "Done. Instances terminating."
}

status() {
  echo "=== Benchmark Instance Status ==="
  for region in "${REGIONS[@]}"; do
    echo "--- $region ---"
    aws ec2 describe-instances --region "$region" \
      --filters "Name=tag:$TAG_KEY,Values=$TAG_VALUE" \
      --query 'Reservations[].Instances[].[InstanceId,State.Name,PublicIpAddress,InstanceType]' \
      --output table
  done
}

# --- Main ---
case "${1:-help}" in
  setup)    setup ;;
  teardown) teardown ;;
  status)   status ;;
  *)        echo "Usage: $0 {setup|teardown|status}" ;;
esac
