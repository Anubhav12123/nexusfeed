variable "sagemaker_training_role_arn" {
  description = "IAM role ARN assumed by the weekly SageMaker training job"
  type        = string
  default     = ""
}

variable "enable_spot_instances_for_training" {
  description = "Use spot instances for the SageMaker training job (blueprint default: true)"
  type        = bool
  default     = true
}
