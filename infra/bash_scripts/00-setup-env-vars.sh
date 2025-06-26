
# --------------  SET VARIABLES --------------------
set -e
export MSYS_NO_PATHCONV=1
# set the name of the resource group
resourceGroup="test-arm-rg"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_dir="$script_dir/../arm_templates"

# set the name and path of the resources

## 1. set the storage account name
storageAccountName="ossa"
storageAccountTemplateFile="$base_dir/storage_account.json"

## 2. set the ai searh service name
aiSearchServiceName="osais"
aiSearchServiceTemplateFile="$base_dir/azure_ai_search.json"

## 3. set the azure speech service name
azureSpeechServiceName="osstt"
azureSpeechServiceRegion="centralindia"
azureSpeechServiceTemplateFile="$base_dir/azure_speech_service.json"

## 4. set the container registry name
containerRegistryName="osacr"
containerRegistryTemplateFile="$base_dir/container_registry.json"

## 5. set the app service PLAN name
aspPremiumName="osaspp"
aspPremiumTemplateFile="$base_dir/app_service_plan_premium.json"

aspBasicName="osaspb"
aspBasicTemplateFile="$base_dir/app_service_plan_basic.json"

## 6. set the azure event hub name and topic name
eventhubName="osevhub"
queryPipelineTopicName="query-eventhub"
ingestionPipelineTopicName="ingestion-eventhub"
eventhubTemplateFile="$base_dir/azure_event_hub.json"

## 7. Azure Openai
azureOpenAIName="osazoai"
azureOpenAITemplateFile="$base_dir/azure_openai.json"

## 8. Managed Identity Name
identityName="osmidentity"

## 9. Docker Images, App service and Container Apps Name
imageTag="1.0"
baseImage="${containerRegistryName}.azurecr.io/osbase:${imageTag}"

mainAppServiceName="osmainapp"
# mainAppTemplateFile="$(realpath "$base_dir/main_app_service.json")"
mainAppTemplateFile="$(cd "$base_dir" && pwd)/main_app_service.json"
# Convert to Windows-style path for Azure CLI
mainAppTemplateFileWin="$(cygpath -w "$mainAppTemplateFile")"
mainAppImageandTag="${containerRegistryName}.azurecr.io/main-app:${imageTag}"

containerAppName="osingestioncons"
containerAppImageAndTag="${containerRegistryName}.azurecr.io/ingestion-consumer:${imageTag}"

# name the container apps environment name
containerAppsEnvName="oscontappenv"

containerRegistry="$containerRegistryName.azurecr.io"