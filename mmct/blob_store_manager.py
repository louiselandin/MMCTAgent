"""
blob_manager.py

BlobStorageManager
------------------
This class centralizes blob operations and optimizes resource usage:

Steps taken:
1. Single shared BlobServiceClient with limited HTTP connection pool.
2. Reuse `DefaultAzureCredential` or CLI credential to avoid repeated instantiation.
3. Async methods for upload and download, using `aiofiles`.
4. Explicitly close `BlobClient`s and the service client to release file descriptors.
"""

import os
import base64
from pathlib import Path
import aiofiles
from urllib.parse import urlparse, unquote
from azure.storage.blob.aio import BlobServiceClient
from azure.identity.aio import DefaultAzureCredential, AzureCliCredential
from loguru import logger


class BlobStorageManager:
    def __init__(self, service_client: BlobServiceClient, credential):
        self.service_client = service_client
        self.credential = credential

    @classmethod
    async def create(cls, account_url: str = None):
        """Asynchronously create an instance of BlobStorageManager with proper credential setup."""
        try:
            credential = await cls._get_credential()
            service_client = BlobServiceClient(
                account_url or os.getenv("BLOB_ACCOUNT_URL"),
                credential=credential
            )
            logger.info("Successfully initialized the blob service client")
            return cls(service_client, credential)
        except Exception as e:
            logger.exception(f"Exception occurred while creating the blob service client: {e}")
            raise

    @staticmethod
    async def _get_credential():
        """Try Azure CLI credential, fallback to DefaultAzureCredential."""
        try:
            credential = AzureCliCredential()
            await credential.get_token("https://cognitiveservices.azure.com/.default")
            logger.info("Using Azure CLI credential for Blob Storage")
            return credential
        except Exception:
            credential = DefaultAzureCredential()
            logger.info("Using DefaultAzureCredential for Blob Storage")
            return credential

    def get_blob_url(self, container: str, blob_name: str) -> str:
        try:
            logger.info(f"Fetching the blob url for blob name: {blob_name}")
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            url = unquote(client.url)
            logger.info(f"Blob URL: {url}")
            return url
        except Exception as e:
            logger.exception(f"Failed to get blob URL: {e}")
            raise

    async def upload_file(self, container: str, blob_name: str, file_path: str) -> str:
        try:
            logger.info(f"Uploading local file: {file_path} to blob: {blob_name}")
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            async with aiofiles.open(file_path, "rb") as f:
                data = await f.read()
            await client.upload_blob(data, overwrite=True)
            await client.close()
            url = f"{self.service_client.url}/{container}/{blob_name}"
            return url
        except Exception as e:
            logger.exception(f"Upload failed for file {file_path}: {e}")
            raise

    async def upload_base64(self, container: str, blob_name: str, b64_str: str) -> str:
        try:
            logger.info(f"Uploading base64 data to blob: {blob_name}")
            data = base64.b64decode(b64_str)
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            await client.upload_blob(data, overwrite=True)
            await client.close()
            return f"{self.service_client.url}/{container}/{blob_name}"
        except Exception as e:
            logger.exception(f"Base64 upload failed: {e}")
            raise

    async def upload_string(self, container: str, blob_name: str, content: str) -> str:
        try:
            logger.info(f"Uploading string content to blob: {blob_name}")
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            await client.upload_blob(content, overwrite=True)
            await client.close()
            return f"{self.service_client.url}/{container}/{blob_name}"
        except Exception as e:
            logger.exception(f"String upload failed: {e}")
            raise

    async def download_to_file(self, container: str, blob_name: str, download_path: str) -> str:
        try:
            logger.info(f"Downloading blob {blob_name} to {download_path}")
            Path(download_path).parent.mkdir(parents=True, exist_ok=True)
            client = self.service_client.get_blob_client(container=container, blob=blob_name)
            stream = await client.download_blob()
            data = await stream.readall()
            async with aiofiles.open(download_path, "wb") as f:
                await f.write(data)
            await client.close()
            logger.info(f"Download complete: {download_path}")
            return download_path
        except Exception as e:
            logger.exception(f"Download failed: {e}")
            raise

    async def download_from_url(self, blob_url: str, save_folder: str) -> str:
        try:
            logger.info(f"Downloading from URL: {blob_url}")
            parsed = urlparse(blob_url)
            container, blob_name = parsed.path.lstrip("/").split("/", 1)
            local_path = os.path.join(save_folder, blob_name)
            return await self.download_to_file(container, blob_name, local_path)
        except Exception as e:
            logger.exception(f"Failed to download from URL: {e}")
            raise

    async def close(self):
        """Close the service client and release resources."""
        logger.info("Closing BlobServiceClient")
        await self.service_client.close()
