# 🚀 MMCT Infrastructure Deployment Guide

This repository provides an automated way to deploy Azure resources and application services for the MMCT Pipelines using **ARM templates**, **Bash scripts**, and a single configuration file (`infra_config.yaml`).

---

## 📋 Prerequisites

Before you begin, ensure the following tools are installed:

### 🔧 Required Tools

| Tool         | Description                              | Link |
|--------------|------------------------------------------|------|
| **Azure CLI**| For Azure login and resource deployment  | [Install Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) |
| **Docker**   | Required for building and pushing images | [Install Docker](https://docs.docker.com/get-docker/) |
| **Bash Shell**| For running the deployment scripts      | Use **Git Bash**, **WSL**, or native **Bash** on Linux/macOS |

### 🔐 Azure Authentication

Authenticate and set your subscription using Azure CLI:

```bash
az login
az account set --subscription "<your-subscription-name-or-id>"
```

### ⚙️ Configuration via infra_config.yaml

The deployment is controlled through a single YAML config file (infra_config.yaml), allowing selective resource provisioning.

### 🔄 How It Works

Each section in the config file controls a part of the deployment:

| Section                    | Purpose                                           |
|---------------------------|---------------------------------------------------|
| deployInfra.*             | Deploy selected Azure infra components via ARM   |
| envCreation.enabled | Create and assign values in the env file     |
| midentityCreation.enabled | Create and assign roles to a Managed Identity     |
| buildAndPushImagesToACR.*         | Build and/or push Docker images of main app and ingestion consumer    |
| deployAppService.enabled  | Deploy producer apps to Azure App Service         |
| deployContainerApps.enabled | Deploy consumer apps to Azure Container Apps  |

Set the values to true or false to enable/disable specific deployments.

### 🚦 Execution Flow

You can use individual scripts or trigger everything through a single orchestrator.

✅ Step-by-Step Breakdown

**Edit Config File**  
Open `infra_config.yaml` and toggle the values according to your needs.

**Make Scripts Executable (if needed)**  
On Linux/macOS/WSL:

```bash
chmod +x *.sh
```

#### **Run the All-in-One Script**

```bash
./deploy-all.sh
```

This will execute the scripts in the following logical order based on config:

1. `00-setup-env-vars.sh`: Sets environment variables.
2. `01-deploy-infra.sh`: Deploys infra resources as per deployInfra section.
3. `02-generate-env.sh`: Generates an env file for the mmct repository for the resources deployed.
4. `03-create-managed-identity.sh`: Creates and assigns roles to Managed Identity.
5. `04-build-push-docker-images.sh`: Builds and pushes Docker images if enabled.
6. `05-deploy-app-service.sh`: Deploys App Services if enabled.
7. `06-deploy-container-app.sh`: Deploys Container Apps if enabled.

### 📁 File Structure

```bash
.
├── infra_config.yaml                    # Main config to control deployments
├── deploy-all.sh                        # Master script   
├── bash_scripts/         
|    ├── 00-setup-env-vars.sh            # Loads variable names
|    ├── 01-deploy-infra.sh              # Deploys infrastructure
|    ├── 02-generate-env.sh              # Handles .env file creation
|    ├── 03-create-managed-identity.sh   # Handles Managed Identity creation and role assignment
|    ├── 04-build-push-docker-images.sh  # Builds/pushes Docker images
|    ├── 05-deploy-app-service.sh        # Deploys App Services
|    └── 06-deploy-container-app.sh      # Deploys Container Apps
└── arm_templates/                  # Contains ARM JSON templates
```

### 💡 Notes

- Always review `infra_config.yaml` before executing.
- Scripts are idempotent – resources won’t be recreated if they already exist.
- If you're using Windows, we recommend using Git Bash or WSL for better compatibility.
