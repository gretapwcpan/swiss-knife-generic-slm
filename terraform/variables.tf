# variables.tf - Variable definitions for DistilBERT SageMaker deployment

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "distilbert"
}

# Training Configuration
variable "training_instance_type" {
  description = "SageMaker instance type for training"
  type        = string
  default     = "ml.p3.8xlarge"  # 4x V100 GPUs - Best value
  
  validation {
    condition = contains([
      "ml.p3.2xlarge",   # 1x V100
      "ml.p3.8xlarge",   # 4x V100 - RECOMMENDED
      "ml.p3.16xlarge",  # 8x V100
      "ml.p4d.24xlarge", # 8x A100
      "ml.g4dn.xlarge",  # 1x T4
      "ml.g4dn.12xlarge", # 4x T4
      "ml.g5.4xlarge",   # 1x A10G
      "ml.g5.12xlarge"   # 4x A10G
    ], var.training_instance_type)
    error_message = "Invalid SageMaker instance type."
  }
}

variable "training_instance_count" {
  description = "Number of training instances"
  type        = number
  default     = 1
}

variable "training_volume_size" {
  description = "EBS volume size in GB for training"
  type        = number
  default     = 100
}

variable "max_training_runtime" {
  description = "Maximum training runtime in seconds (5 days default)"
  type        = number
  default     = 432000  # 5 days
}

variable "use_spot_instances" {
  description = "Use spot instances for training (70% cost savings)"
  type        = bool
  default     = true
}

# Training Hyperparameters (DistilBERT paper defaults)
variable "training_hyperparameters" {
  description = "Hyperparameters for DistilBERT training"
  type        = map(string)
  default = {
    teacher_model               = "bert-base-uncased"
    batch_size                 = "32"      # Per GPU
    gradient_accumulation_steps = "32"     # Effective batch = 4096
    num_epochs                 = "3"
    learning_rate              = "5e-4"    # Paper default
    temperature                = "3.0"     # Paper default
    alpha                      = "0.5"     # Distillation loss weight
    beta                       = "2.0"     # MLM loss weight
    gamma                      = "1.0"     # Cosine loss weight
    mlm_probability           = "0.15"    # Paper default
    max_length                = "512"
    warmup_steps              = "500"
    save_steps                = "1000"
    use_demo_data             = "false"   # Set to "true" for testing
    max_samples               = ""        # Empty for full dataset
  }
}

# Notebook Configuration
variable "create_notebook_instance" {
  description = "Create SageMaker notebook instance for development"
  type        = bool
  default     = true
}

variable "notebook_instance_type" {
  description = "Instance type for notebook"
  type        = string
  default     = "ml.t3.xlarge"
}

variable "github_repo_url" {
  description = "GitHub repository URL for notebook"
  type        = string
  default     = "https://github.com/yourusername/swiss-knife-generic-slm.git"
}

# Network Configuration
variable "enable_network_isolation" {
  description = "Enable network isolation for training"
  type        = bool
  default     = false
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "availability_zones" {
  description = "Availability zones for subnets"
  type        = list(string)
  default     = ["us-east-1a", "us-east-1b"]
}

# Monitoring and Alerts
variable "enable_monitoring" {
  description = "Enable CloudWatch monitoring and alarms"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "cost_alert_threshold" {
  description = "Cost alert threshold in USD"
  type        = number
  default     = 1000
}

variable "alarm_sns_topic_arn" {
  description = "SNS topic ARN for alarms (optional)"
  type        = string
  default     = ""
}

# Data and Code
variable "training_code_path" {
  description = "Path to training code archive"
  type        = string
  default     = "../src.tar.gz"
}

variable "use_custom_container" {
  description = "Use custom Docker container for training"
  type        = bool
  default     = false
}

variable "start_training_job" {
  description = "Automatically start training job"
  type        = bool
  default     = false
}

# Cost Optimization
variable "spot_max_wait" {
  description = "Maximum wait time for spot instances in seconds"
  type        = number
  default     = 432000  # Same as max runtime
}

variable "enable_auto_model_tuning" {
  description = "Enable automatic model tuning (hyperparameter optimization)"
  type        = bool
  default     = false
}

# Tags
variable "additional_tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}
