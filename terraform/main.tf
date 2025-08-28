# main.tf - Main Terraform configuration for DistilBERT SageMaker deployment

terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    # Configure your backend
    # bucket = "your-terraform-state-bucket"
    # key    = "distilbert/terraform.tfstate"
    # region = "us-east-1"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "DistilBERT-Training"
      Environment = var.environment
      ManagedBy   = "Terraform"
      Paper       = "arxiv:1909.10351"
    }
  }
}

# Data sources
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# S3 bucket for training data and model artifacts
resource "aws_s3_bucket" "training_bucket" {
  bucket = "${var.project_name}-${var.environment}-${data.aws_caller_identity.current.account_id}"
  
  tags = {
    Name = "DistilBERT Training Bucket"
  }
}

resource "aws_s3_bucket_versioning" "training_bucket_versioning" {
  bucket = aws_s3_bucket.training_bucket.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "training_bucket_encryption" {
  bucket = aws_s3_bucket.training_bucket.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# S3 bucket for model registry
resource "aws_s3_bucket" "model_registry" {
  bucket = "${var.project_name}-models-${var.environment}-${data.aws_caller_identity.current.account_id}"
  
  tags = {
    Name = "DistilBERT Model Registry"
  }
}

# IAM role for SageMaker
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role-${var.environment}"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
      }
    ]
  })
}

# IAM policy for SageMaker
resource "aws_iam_role_policy" "sagemaker_policy" {
  name = "${var.project_name}-sagemaker-policy"
  role = aws_iam_role.sagemaker_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.training_bucket.arn,
          "${aws_s3_bucket.training_bucket.arn}/*",
          aws_s3_bucket.model_registry.arn,
          "${aws_s3_bucket.model_registry.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "cloudwatch:GetMetricData",
          "cloudwatch:GetMetricStatistics",
          "cloudwatch:ListMetrics"
        ]
        Resource = "*"
      }
    ]
  })
}

# Attach AWS managed policy for SageMaker
resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# VPC for SageMaker (optional, for network isolation)
resource "aws_vpc" "sagemaker_vpc" {
  count = var.enable_network_isolation ? 1 : 0
  
  cidr_block           = var.vpc_cidr
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${var.project_name}-vpc-${var.environment}"
  }
}

# Subnets for SageMaker
resource "aws_subnet" "sagemaker_subnet" {
  count = var.enable_network_isolation ? length(var.availability_zones) : 0
  
  vpc_id            = aws_vpc.sagemaker_vpc[0].id
  cidr_block        = cidrsubnet(var.vpc_cidr, 4, count.index)
  availability_zone = var.availability_zones[count.index]
  
  tags = {
    Name = "${var.project_name}-subnet-${count.index}-${var.environment}"
  }
}

# Security group for SageMaker
resource "aws_security_group" "sagemaker_sg" {
  count = var.enable_network_isolation ? 1 : 0
  
  name        = "${var.project_name}-sg-${var.environment}"
  description = "Security group for DistilBERT SageMaker training"
  vpc_id      = aws_vpc.sagemaker_vpc[0].id
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.project_name}-sg-${var.environment}"
  }
}

# CloudWatch Log Group for training jobs
resource "aws_cloudwatch_log_group" "training_logs" {
  name              = "/aws/sagemaker/${var.project_name}-${var.environment}"
  retention_in_days = var.log_retention_days
  
  tags = {
    Name = "DistilBERT Training Logs"
  }
}

# ECR repository for custom training container (optional)
resource "aws_ecr_repository" "training_container" {
  count = var.use_custom_container ? 1 : 0
  
  name                 = "${var.project_name}-training-${var.environment}"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  tags = {
    Name = "DistilBERT Training Container"
  }
}

# SageMaker notebook instance for development
resource "aws_sagemaker_notebook_instance" "dev_notebook" {
  count = var.create_notebook_instance ? 1 : 0
  
  name                  = "${var.project_name}-notebook-${var.environment}"
  instance_type         = var.notebook_instance_type
  role_arn             = aws_iam_role.sagemaker_role.arn
  default_code_repository = var.github_repo_url
  
  tags = {
    Name = "DistilBERT Development Notebook"
  }
}

# Upload training code to S3
resource "aws_s3_object" "training_code" {
  bucket = aws_s3_bucket.training_bucket.id
  key    = "code/training.tar.gz"
  source = var.training_code_path
  etag   = filemd5(var.training_code_path)
}

# Upload training script
resource "aws_s3_object" "training_script" {
  bucket  = aws_s3_bucket.training_bucket.id
  key     = "code/sagemaker_train.py"
  content = file("${path.module}/scripts/sagemaker_train.py")
  etag    = md5(file("${path.module}/scripts/sagemaker_train.py"))
}

# SageMaker training job
resource "null_resource" "training_job" {
  count = var.start_training_job ? 1 : 0
  
  triggers = {
    training_config = jsonencode(var.training_hyperparameters)
    code_version    = aws_s3_object.training_code.etag
  }
  
  provisioner "local-exec" {
    command = templatefile("${path.module}/scripts/start_training.sh", {
      job_name            = "${var.project_name}-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
      role_arn           = aws_iam_role.sagemaker_role.arn
      instance_type      = var.training_instance_type
      instance_count     = var.training_instance_count
      volume_size        = var.training_volume_size
      max_runtime        = var.max_training_runtime
      use_spot_instances = var.use_spot_instances
      hyperparameters    = jsonencode(var.training_hyperparameters)
      input_data_s3      = "s3://${aws_s3_bucket.training_bucket.id}/data/"
      output_data_s3     = "s3://${aws_s3_bucket.training_bucket.id}/models/"
      code_s3            = "s3://${aws_s3_bucket.training_bucket.id}/code/training.tar.gz"
    })
  }
}

# CloudWatch Alarms
resource "aws_cloudwatch_metric_alarm" "gpu_utilization" {
  count = var.enable_monitoring ? 1 : 0
  
  alarm_name          = "${var.project_name}-gpu-utilization-${var.environment}"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = "2"
  metric_name        = "GPUUtilization"
  namespace          = "/aws/sagemaker/TrainingJobs"
  period             = "300"
  statistic          = "Average"
  threshold          = "50"
  alarm_description  = "This metric monitors GPU utilization"
  alarm_actions      = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []
  
  dimensions = {
    TrainingJobName = "${var.project_name}-*"
  }
}

resource "aws_cloudwatch_metric_alarm" "training_cost" {
  count = var.enable_monitoring ? 1 : 0
  
  alarm_name          = "${var.project_name}-training-cost-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name        = "EstimatedCharges"
  namespace          = "AWS/Billing"
  period             = "86400"
  statistic          = "Maximum"
  threshold          = var.cost_alert_threshold
  alarm_description  = "Alert when training costs exceed threshold"
  alarm_actions      = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []
  
  dimensions = {
    Currency = "USD"
  }
}

# Outputs
output "s3_training_bucket" {
  value       = aws_s3_bucket.training_bucket.id
  description = "S3 bucket for training data and artifacts"
}

output "s3_model_registry" {
  value       = aws_s3_bucket.model_registry.id
  description = "S3 bucket for model registry"
}

output "sagemaker_role_arn" {
  value       = aws_iam_role.sagemaker_role.arn
  description = "IAM role ARN for SageMaker"
}

output "notebook_instance_name" {
  value       = var.create_notebook_instance ? aws_sagemaker_notebook_instance.dev_notebook[0].name : ""
  description = "Name of the SageMaker notebook instance"
}

output "training_job_command" {
  value = <<-EOT
    aws sagemaker create-training-job \
      --training-job-name ${var.project_name}-$(date +%Y%m%d-%H%M%S) \
      --role-arn ${aws_iam_role.sagemaker_role.arn} \
      --algorithm-specification TrainingImage=763104351884.dkr.ecr.${data.aws_region.current.name}.amazonaws.com/pytorch-training:2.0.0-gpu-py310,TrainingInputMode=File \
      --resource-config InstanceType=${var.training_instance_type},InstanceCount=${var.training_instance_count},VolumeSizeInGB=${var.training_volume_size} \
      --input-data-config '[{"ChannelName":"training","DataSource":{"S3DataSource":{"S3DataType":"S3Prefix","S3Uri":"s3://${aws_s3_bucket.training_bucket.id}/data/","S3DataDistributionType":"FullyReplicated"}}}]' \
      --output-data-config S3OutputPath=s3://${aws_s3_bucket.training_bucket.id}/models/ \
      --hyper-parameters '${jsonencode(var.training_hyperparameters)}' \
      --stopping-condition MaxRuntimeInSeconds=${var.max_training_runtime}
  EOT
  description = "AWS CLI command to start training job"
}
