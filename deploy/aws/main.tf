# ============================================================================
# Octopus BFT — EC2 1000-Node Consensus Deployment
# ============================================================================
# Production infrastructure for the 1000-replica WAN benchmark reported in §VI.
# Provisions 100 EC2 c5.xlarge instances (4 vCPU, 8 GiB RAM) across
# 2 availability zones. Each VM runs 10 Octopus replicas = 1000 total.
# NetEm applies 40ms one-way delay (80ms RTT) to match WAN conditions.
#
# Usage:
#   cd deploy/aws
#   terraform init
#   terraform plan -var="key_name=your-ec2-key"
#   terraform apply -var="key_name=your-ec2-key"
#
# After benchmark:
#   terraform destroy
# ============================================================================

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# ── Variables ───────────────────────────────────────────────────────────────

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "key_name" {
  description = "EC2 SSH key pair name"
  type        = string
}

variable "num_vms" {
  description = "Number of EC2 instances (each runs replicas_per_vm Octopus nodes)"
  type        = number
  default     = 100
}

variable "replicas_per_vm" {
  description = "Octopus replicas per VM"
  type        = number
  default     = 10
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "c5.xlarge" # 4 vCPU, 8 GiB
}

variable "ami_id" {
  description = "AMI ID (Ubuntu 22.04 arm64 or amd64). Leave empty for auto-lookup."
  type        = string
  default     = ""
}

variable "wan_delay_ms" {
  description = "Simulated WAN delay via NetEm (one-way, ms). 0 = no shaping."
  type        = number
  default     = 40 # 80ms RTT
}

variable "bandwidth_mbps" {
  description = "Bandwidth limit per VM (Mbps). 0 = no shaping."
  type        = number
  default     = 0
}

variable "consensus_instances" {
  description = "Number of parallel consensus lanes (m)"
  type        = number
  default     = 10
}

variable "batch_txs" {
  description = "Transactions per proposal batch"
  type        = number
  default     = 8192 # 512KB / 64B = 8192
}

variable "allowed_ssh_cidr" {
  description = "CIDR allowed for SSH access"
  type        = string
  default     = "0.0.0.0/0"
}

variable "ecr_repo_uri" {
  description = "ECR repository URI for the Octopus Docker image (set by orchestrator script)"
  type        = string
  default     = ""
}

variable "s3_manifest_bucket" {
  description = "S3 bucket name containing node manifests (set by orchestrator script)"
  type        = string
  default     = ""
}

# ── Data Sources ────────────────────────────────────────────────────────────

data "aws_ami" "ubuntu" {
  count       = var.ami_id == "" ? 1 : 0
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

locals {
  ami            = var.ami_id != "" ? var.ami_id : data.aws_ami.ubuntu[0].id
  total_replicas = var.num_vms * var.replicas_per_vm
  azs            = ["${var.aws_region}a", "${var.aws_region}b"]
}

# ── Networking ──────────────────────────────────────────────────────────────

resource "aws_vpc" "octopus" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = { Name = "octopus-benchmark-vpc" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.octopus.id
  tags   = { Name = "octopus-igw" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.octopus.id
  cidr_block              = "10.0.${count.index}.0/24"
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = { Name = "octopus-subnet-${count.index}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.octopus.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }

  tags = { Name = "octopus-rt" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ── Security Group ──────────────────────────────────────────────────────────

resource "aws_security_group" "octopus" {
  name        = "octopus-benchmark-sg"
  description = "Allow SSH + intra-cluster traffic"
  vpc_id      = aws_vpc.octopus.id

  # SSH from allowed CIDR
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # All intra-VPC traffic (P2P + HTTP)
  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  ingress {
    from_port   = 0
    to_port     = 65535
    protocol    = "udp"
    cidr_blocks = ["10.0.0.0/16"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "octopus-sg" }
}

# ── EC2 Instances ───────────────────────────────────────────────────────────

resource "aws_instance" "octopus" {
  count                  = var.num_vms
  ami                    = local.ami
  instance_type          = var.instance_type
  key_name               = var.key_name
  subnet_id              = aws_subnet.public[count.index % 2].id
  vpc_security_group_ids = [aws_security_group.octopus.id]

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    vm_index         = count.index
    replicas_per_vm  = var.replicas_per_vm
    total_replicas   = local.total_replicas
    instances        = var.consensus_instances
    batch_txs        = var.batch_txs
    wan_delay_ms     = var.wan_delay_ms
    bandwidth_mbps   = var.bandwidth_mbps
    ecr_repo_uri     = var.ecr_repo_uri
    s3_manifest_bucket = var.s3_manifest_bucket
  })

  tags = {
    Name    = "octopus-node-${count.index}"
    Project = "octopus-benchmark"
    Role    = "consensus"
  }
}

# ── Outputs ─────────────────────────────────────────────────────────────────

output "instance_ips" {
  description = "Public IPs of all benchmark VMs"
  value       = aws_instance.octopus[*].public_ip
}

output "private_ips" {
  description = "Private IPs (used for intra-cluster P2P)"
  value       = aws_instance.octopus[*].private_ip
}

output "total_replicas" {
  value = local.total_replicas
}

output "ssh_command" {
  value = "ssh -i ~/.ssh/${var.key_name}.pem ubuntu@${try(aws_instance.octopus[0].public_ip, "pending")}"
}
