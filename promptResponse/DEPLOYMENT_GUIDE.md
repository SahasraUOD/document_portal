# Deployment Guide for Document Portal

A comprehensive step-by-step guide for pushing the Document Portal project to Git and deploying to AWS ECS (Fargate) with ECR container registry.

---

## 1. Pre-Deployment Checklist

### 1.1 Required Tools
- [ ] **Git** - Version control (`git --version`)
- [ ] **Docker** - Container runtime (`docker --version`)
- [ ] **AWS CLI v2** - AWS command line tool (`aws --version`)
- [ ] **Python 3.10+** - For local testing (`python --version`)

### 1.2 Required Accounts & Access
- [ ] **GitHub Account** - For repository hosting
- [ ] **AWS Account** - For deployment infrastructure
- [ ] **AWS IAM User** - With programmatic access credentials
- [ ] **API Keys** - GOOGLE_API_KEY, GROQ_API_KEY, LANGCHAIN_API_KEY, HF_TOKEN

### 1.3 Environment Variables (Local Development)
Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
HF_TOKEN=your_huggingface_token
LLM_PROVIDER=google
ENV=local
```

> ⚠️ **SECURITY WARNING**: Never commit `.env` or any file containing API keys to Git!

### 1.4 Configuration Verification
- [ ] Verify `config/config.yaml` has correct model settings
- [ ] Verify `requirements.txt` has all dependencies
- [ ] Verify `Dockerfile` exposes port 8080

---

## 2. Git Setup & Code Push

### Step 2.1: Initialize Git Repository
```bash
cd document_portal
git init
```

### Step 2.2: Create/Verify .gitignore
Ensure these entries exist in `.gitignore`:
```gitignore
# Environment & Secrets
.env
.env.*
*.pem
*credentials*

# Python
__pycache__/
*.py[cod]
*$py.class
.Python
*.so
.venv/
venv/
ENV/

# IDE
.vscode/
.idea/

# Build
dist/
build/
*.egg-info/

# Data & Logs
logs/
data/
*.log

# OS
.DS_Store
Thumbs.db
```

### Step 2.3: Add Files and Initial Commit
```bash
git add .
git commit -m "Initial commit: Document Portal - FastAPI LLM application"
```

### Step 2.4: Create GitHub Repository
1. Go to [GitHub](https://github.com) → New Repository
2. Repository name: `document-portal`
3. Keep it **Private** (contains deployment configs)
4. Do NOT initialize with README (we have local code)

### Step 2.5: Add Remote and Push
```bash
git remote add origin https://github.com/[YOUR_USERNAME]/document-portal.git
git branch -M main
git push -u origin main
```

### Step 2.6: Add GitHub Secrets (for CI/CD)
Navigate to: **Repository → Settings → Secrets and variables → Actions**

Add these repository secrets:
| Secret Name | Value |
|-------------|-------|
| `AWS_ACCESS_KEY_ID` | Your IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | Your IAM user secret key |
| `AWS_REGION` | `us-east-1` (or your preferred region) |

---

## 3. AWS Infrastructure Setup

### Step 3.1: Configure AWS CLI
```bash
aws configure
# Enter:
# AWS Access Key ID: [YOUR_ACCESS_KEY]
# AWS Secret Access Key: [YOUR_SECRET_KEY]
# Default region: us-east-1
# Default output format: json
```

### Step 3.2: Store API Keys in AWS Secrets Manager
```bash
# Store GOOGLE_API_KEY
aws secretsmanager create-secret \
    --name GOOGLE_API_KEY \
    --secret-string "your_google_api_key" \
    --region us-east-1

# Store GROQ_API_KEY
aws secretsmanager create-secret \
    --name GROQ_API_KEY \
    --secret-string "your_groq_api_key" \
    --region us-east-1

# Store LANGCHAIN_API_KEY
aws secretsmanager create-secret \
    --name LANGCHAIN_API_KEY \
    --secret-string "your_langchain_api_key" \
    --region us-east-1

# Store HF_TOKEN
aws secretsmanager create-secret \
    --name HF_TOKEN \
    --secret-string "your_hf_token" \
    --region us-east-1
```

### Step 3.3: Create IAM Policies
Create two custom IAM policies for your IAM user/role:

**Policy 1: AllowECSLogs**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "Statement1",
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents"
            ],
            "Resource": "*"
        }
    ]
}
```

**Policy 2: AllowSecretsAccess**
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "Statement1",
            "Effect": "Allow",
            "Action": "secretsmanager:GetSecretValue",
            "Resource": "arn:aws:secretsmanager:us-east-1:*:secret:*"
        }
    ]
}
```

To create via AWS Console:
1. IAM → Policies → Create policy
2. JSON tab → Paste policy
3. Name: `AllowECSLogs` and `AllowSecretsAccess`
4. Attach both policies to your IAM user or ECS task execution role

### Step 3.4: Create ECR Repository
```bash
aws ecr create-repository \
    --repository-name documentportal \
    --image-scanning-configuration scanOnPush=true \
    --region us-east-1
```

Note the repository URI:
```
[ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/documentportal
```

---

## 4. Build & Push Docker Image

### Step 4.1: Build Docker Image Locally
```bash
cd document_portal
docker build -t documentportal:latest .
```

### Step 4.2: Test Locally (Optional but Recommended)
```bash
docker run -d -p 8080:8080 \
    -e GOOGLE_API_KEY=your_key \
    -e GROQ_API_KEY=your_key \
    -e ENV=local \
    documentportal:latest

# Test the application
curl http://localhost:8080/health
# Expected: {"status": "ok"}

# Stop container
docker stop $(docker ps -q --filter ancestor=documentportal:latest)
```

### Step 4.3: Authenticate Docker with ECR
```bash
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com
```

### Step 4.4: Tag and Push Image
```bash
# Tag the image
docker tag documentportal:latest \
    [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/documentportal:latest

# Push to ECR
docker push [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/documentportal:latest
```

---

## 5. Deploy to AWS ECS (Fargate)

### Step 5.1: Deploy CloudFormation Stack
```bash
aws cloudformation create-stack \
    --stack-name document-portal-stack \
    --template-body file://infrastructure/document-portal-cf.yaml \
    --parameters ParameterKey=ImageUrl,ParameterValue=[ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/documentportal:latest \
    --capabilities CAPABILITY_NAMED_IAM \
    --region us-east-1
```

### Step 5.2: Monitor Stack Creation
```bash
# Check status
aws cloudformation describe-stacks \
    --stack-name document-portal-stack \
    --query 'Stacks[0].StackStatus' \
    --region us-east-1

# Wait for completion (takes ~5-10 minutes)
aws cloudformation wait stack-create-complete \
    --stack-name document-portal-stack \
    --region us-east-1
```

### Step 5.3: Configure Security Group Inbound Rules
Via AWS Console:
1. Go to **EC2 → Security Groups**
2. Find the security group created by CloudFormation
3. Edit **Inbound rules** → Add rule:
   - Type: `Custom TCP`
   - Port range: `8080`
   - Source: `0.0.0.0/0` (or restrict to your IP)

---

## 6. Verification & Testing

### Step 6.1: Get Public IP Address
**Option A: Via AWS Console**
1. Go to **ECS → Clusters → document-portal-cluster**
2. Click on **document-portal-service**
3. Click on **Tasks** tab
4. Click on the running task
5. Find **Public IP** in the Network section

**Option B: If Public IP not visible**
1. Click on the **ENI ID** (Elastic Network Interface)
2. Find **Public IPv4 address**

### Step 6.2: Test Health Endpoint
```bash
curl http://[PUBLIC_IP]:8080/health
# Expected: {"status": "ok"}
```

### Step 6.3: Test Web Interface
Open in browser:
```
http://[PUBLIC_IP]:8080
```

### Step 6.4: Test API Endpoints
```bash
# Test document analysis (requires file upload)
curl -X POST "http://[PUBLIC_IP]:8080/analyze" \
    -F "files=@sample.pdf"

# Test chat index
curl -X POST "http://[PUBLIC_IP]:8080/chat/index" \
    -F "files=@document.pdf"
```

### Step 6.5: Monitor Logs
**Via AWS Console:**
1. Go to **CloudWatch → Log groups**
2. Find `/ecs/documentportal`
3. Click on the latest log stream
4. Review application logs

**Via AWS CLI:**
```bash
aws logs tail /ecs/documentportal --follow --region us-east-1
```

---

## 7. Update Deployment

### Step 7.1: Push Code Changes
```bash
git add .
git commit -m "feat: your changes description"
git push origin main
```

### Step 7.2: Rebuild and Push Docker Image
```bash
# Rebuild
docker build -t documentportal:latest .

# Re-authenticate (if token expired)
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag documentportal:latest \
    [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/documentportal:latest
docker push [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/documentportal:latest
```

### Step 7.3: Force New Deployment
```bash
aws ecs update-service \
    --cluster document-portal-cluster \
    --service document-portal-service \
    --force-new-deployment \
    --region us-east-1
```

---

## 8. Rollback Procedures

### Step 8.1: Rollback to Previous Image
```bash
# List available images
aws ecr describe-images \
    --repository-name documentportal \
    --region us-east-1

# Create new task definition with previous image tag
# Then update service to use that task definition
```

### Step 8.2: Delete Stack and Redeploy (Full Rollback)
```bash
# Delete current stack
aws cloudformation delete-stack \
    --stack-name document-portal-stack \
    --region us-east-1

# Wait for deletion
aws cloudformation wait stack-delete-complete \
    --stack-name document-portal-stack \
    --region us-east-1

# Redeploy with known-good image
aws cloudformation create-stack \
    --stack-name document-portal-stack \
    --template-body file://infrastructure/document-portal-cf.yaml \
    --parameters ParameterKey=ImageUrl,ParameterValue=[ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/documentportal:[PREVIOUS_TAG] \
    --capabilities CAPABILITY_NAMED_IAM \
    --region us-east-1
```

---

## 9. Troubleshooting

### Issue: Container fails to start
**Symptoms:** Task keeps restarting, no public IP assigned
**Solutions:**
1. Check CloudWatch logs for error messages
2. Verify secrets are accessible (IAM permissions)
3. Test Docker image locally first
4. Verify port 8080 is exposed in Dockerfile

### Issue: Cannot access application
**Symptoms:** Connection timeout on `http://[IP]:8080`
**Solutions:**
1. Verify Security Group allows inbound TCP 8080
2. Check task is in `RUNNING` state
3. Verify `AssignPublicIp: ENABLED` in service config

### Issue: Secrets not injected
**Symptoms:** Application crashes with "Missing API keys"
**Solutions:**
1. Verify secrets exist in Secrets Manager
2. Check IAM role has `secretsmanager:GetSecretValue` permission
3. Verify secret ARNs in CloudFormation match actual ARNs

### Issue: CloudFormation stack creation fails
**Symptoms:** Stack status shows `ROLLBACK_COMPLETE`
**Solutions:**
```bash
# View failure reason
aws cloudformation describe-stack-events \
    --stack-name document-portal-stack \
    --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`]' \
    --region us-east-1

# Delete failed stack before retrying
aws cloudformation delete-stack \
    --stack-name document-portal-stack \
    --region us-east-1
```

### Issue: Image push fails
**Symptoms:** `denied: Your authorization token has expired`
**Solutions:**
```bash
# Re-authenticate with ECR
aws ecr get-login-password --region us-east-1 | \
    docker login --username AWS --password-stdin \
    [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com
```

---

## 10. Cost & Cleanup

### Estimated Costs (us-east-1)
| Resource | Configuration | Est. Monthly Cost |
|----------|--------------|-------------------|
| ECS Fargate | 1 vCPU, 8GB RAM | ~$40-50 |
| ECR Storage | <1GB | ~$0.10 |
| CloudWatch Logs | ~1GB/month | ~$0.50 |
| Secrets Manager | 4 secrets | ~$1.60 |
| **Total** | | **~$45-55/month** |

### Cleanup All Resources
```bash
# Delete CloudFormation stack (removes ECS, VPC, etc.)
aws cloudformation delete-stack \
    --stack-name document-portal-stack \
    --region us-east-1

# Delete ECR repository
aws ecr delete-repository \
    --repository-name documentportal \
    --force \
    --region us-east-1

# Delete secrets (optional - may want to keep these)
aws secretsmanager delete-secret \
    --secret-id GOOGLE_API_KEY \
    --force-delete-without-recovery \
    --region us-east-1
```

---

## Quick Reference

### Key URLs
- Application: `http://[PUBLIC_IP]:8080`
- Health Check: `http://[PUBLIC_IP]:8080/health`
- ECS Console: `https://console.aws.amazon.com/ecs`
- CloudWatch Logs: `https://console.aws.amazon.com/cloudwatch`

### Key Commands
```bash
# Build & Push
docker build -t documentportal:latest .
docker push [ACCOUNT_ID].dkr.ecr.us-east-1.amazonaws.com/documentportal:latest

# Deploy
aws ecs update-service --cluster document-portal-cluster --service document-portal-service --force-new-deployment --region us-east-1

# Logs
aws logs tail /ecs/documentportal --follow --region us-east-1

# Status
aws ecs describe-services --cluster document-portal-cluster --services document-portal-service --region us-east-1
```

---

*Last Updated: February 2026*
*Target Platform: AWS ECS Fargate with ECR*
*Application: Document Portal - FastAPI LLM Application*
