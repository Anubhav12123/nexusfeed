terraform {
  required_version = ">= 1.7"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  default = "us-east-1"
}

variable "environment" {
  default = "production"
}

# --- EKS: hosts the API + worker deployments ---
module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  version         = "~> 20.0"
  cluster_name    = "nexusfeed-${var.environment}"
  cluster_version = "1.30"

  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets

  eks_managed_node_groups = {
    api_nodes = {
      instance_types = ["m6i.large"]
      min_size       = 2
      max_size       = 20
      desired_size   = 2
    }
  }
}

# --- RDS: PostgreSQL with pgvector for item embeddings + audit interactions ---
resource "aws_db_instance" "nexusfeed_postgres" {
  identifier           = "nexusfeed-${var.environment}"
  engine               = "postgres"
  engine_version       = "16.4"
  instance_class       = "db.r6g.large"
  allocated_storage    = 100
  storage_type         = "gp3"
  db_name              = "nexusfeed"
  username             = "nexusfeed"
  manage_master_user_password = true
  multi_az             = true
  backup_retention_period = 7
  deletion_protection  = true
}

# --- ElastiCache: Redis Cluster for the online feature store ---
resource "aws_elasticache_replication_group" "nexusfeed_redis" {
  replication_group_id       = "nexusfeed-${var.environment}"
  description                = "NexusFeed online feature store"
  engine                     = "redis"
  engine_version             = "7.1"
  node_type                  = "cache.r6g.large"
  num_cache_clusters         = 3
  automatic_failover_enabled = true
  parameter_group_name       = "default.redis7.cluster.on"
}

# --- MSK: managed Kafka, 3 brokers, replication factor 3 ---
resource "aws_msk_cluster" "nexusfeed_kafka" {
  cluster_name           = "nexusfeed-${var.environment}"
  kafka_version          = "3.7.x"
  number_of_broker_nodes = 3

  broker_node_group_info {
    instance_type   = "kafka.m5.large"
    client_subnets  = module.vpc.private_subnets
    storage_info {
      ebs_storage_info { volume_size = 500 }
    }
  }

  configuration_info {
    arn      = aws_msk_configuration.nexusfeed.arn
    revision = aws_msk_configuration.nexusfeed.latest_revision
  }
}

resource "aws_msk_configuration" "nexusfeed" {
  name              = "nexusfeed-${var.environment}"
  kafka_versions    = ["3.7.x"]
  server_properties = <<EOF
default.replication.factor=3
min.insync.replicas=2
num.partitions=24
EOF
}

# --- S3: item content embeddings, Parquet training snapshots, model artifacts ---
resource "aws_s3_bucket" "nexusfeed_artifacts" {
  bucket = "nexusfeed-artifacts-${var.environment}"
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  name    = "nexusfeed-${var.environment}"
  cidr    = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = true
}

output "eks_cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "postgres_endpoint" {
  value = aws_db_instance.nexusfeed_postgres.endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_replication_group.nexusfeed_redis.primary_endpoint_address
}

output "kafka_bootstrap_brokers" {
  value = aws_msk_cluster.nexusfeed_kafka.bootstrap_brokers
}
