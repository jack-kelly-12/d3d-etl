import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv


def download_s3_bucket(bucket_name, local_directory='./downloads', aws_access_key=None, aws_secret_key=None, region='us-east-1'):
    """
    Download all files from an S3 bucket to a local directory
    
    Args:
        bucket_name (str): Name of the S3 bucket
        local_directory (str): Local directory to save files (default: './downloads')
        aws_access_key (str): AWS access key (optional if using AWS CLI/credentials)
        aws_secret_key (str): AWS secret key (optional if using AWS CLI/credentials)
        region (str): AWS region (default: 'us-east-1')
    """

    # Create S3 client
    try:
        if aws_access_key and aws_secret_key:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=region
            )
        else:
            # Use default credentials (from AWS CLI, environment, or IAM role)
            s3_client = boto3.client('s3', region_name=region)

    except NoCredentialsError:
        print("Error: AWS credentials not found. Please configure your credentials.")
        return False

    # Create local directory if it doesn't exist
    Path(local_directory).mkdir(parents=True, exist_ok=True)

    try:
        # List all objects in the bucket
        print(f"Fetching list of objects from bucket: {bucket_name}")

        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name)

        total_files = 0
        downloaded_files = 0
        failed_files = []

        for page in pages:
            if 'Contents' not in page:
                print("Bucket is empty or no objects found.")
                return True

            for obj in page['Contents']:
                object_key = obj['Key']
                total_files += 1

                # Skip if it's a "folder" (ends with /)
                if object_key.endswith('/'):
                    print(f"Skipping folder: {object_key}")
                    continue

                # Create local file path
                local_file_path = os.path.join(local_directory, object_key)

                # Create local directories if they don't exist
                local_file_dir = os.path.dirname(local_file_path)
                if local_file_dir:
                    Path(local_file_dir).mkdir(parents=True, exist_ok=True)

                try:
                    # Download the file
                    print(f"Downloading: {object_key} -> {local_file_path}")
                    s3_client.download_file(bucket_name, object_key, local_file_path)
                    downloaded_files += 1

                except ClientError as e:
                    error_msg = f"Failed to download {object_key}: {e}"
                    print(error_msg)
                    failed_files.append(object_key)

                except Exception as e:
                    error_msg = f"Unexpected error downloading {object_key}: {e}"
                    print(error_msg)
                    failed_files.append(object_key)

        # Print summary
        print("\n--- Download Summary ---")
        print(f"Total objects found: {total_files}")
        print(f"Successfully downloaded: {downloaded_files}")
        print(f"Failed downloads: {len(failed_files)}")

        if failed_files:
            print("\nFailed files:")
            for file in failed_files:
                print(f"  - {file}")

        return len(failed_files) == 0

    except ClientError as e:
        print(f"Error accessing bucket {bucket_name}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Configuration
    BUCKET_NAME = 'd3-dashboard-kellyjc'
    LOCAL_DIRECTORY = '../data'  # Change this to your desired local path

    # Get credentials from environment variables
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-1')

    if aws_access_key and aws_secret_key:
        print("Starting download using credentials from .env file...")
        success = download_s3_bucket(BUCKET_NAME, LOCAL_DIRECTORY, aws_access_key, aws_secret_key, aws_region)
    else:
        print("Starting download using default AWS credentials...")
        success = download_s3_bucket(BUCKET_NAME, LOCAL_DIRECTORY)

    if success:
        print("\n✅ Download completed successfully!")
        print(f"Files saved to: {os.path.abspath(LOCAL_DIRECTORY)}")
    else:
        print("\n❌ Download completed with errors.")

if __name__ == "__main__":
    main()
