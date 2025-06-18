#!/bin/bash
set -e

CONFIG_FILE="./infra_config.yaml"
bashScriptsDirName="bash_scripts"

get_yaml_value() {
    python -c "import yaml, sys; print(yaml.safe_load(open('$CONFIG_FILE'))$1)"
}

# 1. deploy infrastructure
if [[ "$(get_yaml_value "['deployInfra']['enabled']")" == "True" ]]; then
    echo "==> Running Infra Deployment..."
    bash ./$bashScriptsDirName/01-deploy-infra.sh
else
    echo "=> Skipping infrastructure deployment...!"
fi

# 2. Creating managed identity resource
if [[ "$(get_yaml_value "['midentityCreation']['enabled']")" == "True" ]]; then
    echo "==> Creating Managed Identity Resource..."
    bash ./$bashScriptsDirName/02-create-managed-identity.sh
else
    echo "=> Skipping managed identity resource creation...!"
fi

# 3. building and pushing images
if [[ "$(get_yaml_value "['buildPushImages']['enabled']")" == "True" ]]; then
    echo "==> Building and pushing images..."
    bash ./$bashScriptsDirName/03-build-push-images.sh
else
    echo "=> Skipping building and pushing of docker images...!"
fi


# 4. deploy app services
if [[ "$(get_yaml_value "['deployAppService']['enabled']")" == "True" ]]; then
    echo "==> Deploying App Services..."
    bash ./$bashScriptsDirName/04-deploy-app-service.sh
else
    echo "=> Skipping the deployment of app services...!"
fi

# 5. deploy container apps
if [[ "$(get_yaml_value "['deployContainerApps']['enabled']")" == "True" ]]; then
    echo "==> Deploy container apps..."
    bash ./$bashScriptsDirName/05-deploy-container-app.sh
else
    echo "=> Skipping the deployment of container apps...!"
fi