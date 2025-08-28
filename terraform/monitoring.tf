# monitoring.tf - TensorBoard & MLflow configuration for DistilBERT training

# MLflow Server on Fargate
resource "aws_ecs_cluster" "mlflow" {
  count = var.enable_mlflow ? 1 : 0
  name  = "${var.project_name}-mlflow-${var.environment}"
}

resource "aws_ecs_task_definition" "mlflow" {
  count                    = var.enable_mlflow ? 1 : 0
  family                   = "${var.project_name}-mlflow"
  network_mode            = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                     = "1024"
  memory                  = "2048"
  execution_role_arn      = aws_iam_role.sagemaker_role.arn

  container_definitions = jsonencode([{
    name  = "mlflow"
    image = "ghcr.io/mlflow/mlflow:latest"
    command = [
      "mlflow", "server",
      "--host", "0.0.0.0",
      "--port", "5000",
      "--backend-store-uri", "sqlite:///mlflow.db",
      "--default-artifact-root", "s3://${aws_s3_bucket.model_registry.id}/mlflow"
    ]
    portMappings = [{
      containerPort = 5000
      protocol      = "tcp"
    }]
    logConfiguration = {
      logDriver = "awslogs"
      options = {
        "awslogs-group"         = "/ecs/mlflow"
        "awslogs-region"        = var.aws_region
        "awslogs-stream-prefix" = "mlflow"
      }
    }
  }])
}

# TensorBoard on SageMaker
resource "aws_sagemaker_app" "tensorboard" {
  count        = var.enable_tensorboard ? 1 : 0
  domain_id    = aws_sagemaker_domain.main[0].id
  app_name     = "${var.project_name}-tensorboard"
  app_type     = "TensorBoard"
  
  resource_spec {
    instance_type = "ml.t3.medium"
  }
}

# SageMaker Domain for TensorBoard
resource "aws_sagemaker_domain" "main" {
  count       = var.enable_tensorboard ? 1 : 0
  domain_name = "${var.project_name}-domain"
  auth_mode   = "IAM"
  vpc_id      = var.enable_network_isolation ? aws_vpc.sagemaker_vpc[0].id : data.aws_vpc.default.id
  subnet_ids  = var.enable_network_isolation ? aws_subnet.sagemaker_subnet[*].id : data.aws_subnets.default.ids

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_role.arn
    tensor_board_app_settings {
      default_resource_spec {
        instance_type = "ml.t3.medium"
      }
    }
  }
}

# ALB for MLflow
resource "aws_lb" "mlflow" {
  count              = var.enable_mlflow ? 1 : 0
  name               = "${var.project_name}-mlflow-lb"
  internal           = false
  load_balancer_type = "application"
  subnets           = var.enable_network_isolation ? aws_subnet.sagemaker_subnet[*].id : data.aws_subnets.default.ids
}

# Outputs
output "mlflow_url" {
  value = var.enable_mlflow ? "http://${aws_lb.mlflow[0].dns_name}:5000" : ""
}

output "tensorboard_command" {
  value = var.enable_tensorboard ? "aws sagemaker describe-app --domain-id ${aws_sagemaker_domain.main[0].id} --app-type TensorBoard --app-name ${var.project_name}-tensorboard" : ""
}

# Data sources for default VPC
data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Variables for monitoring
variable "enable_tensorboard" {
  default = true
}

variable "enable_mlflow" {
  default = true
}
