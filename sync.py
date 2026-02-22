import os
import argparse
from minio import Minio

import dotenv
dotenv.load_dotenv()

BUCKET_NAME = "sign"
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
client = Minio(
    endpoint=os.getenv("R2_ENDPOINT", ""),
    access_key=os.getenv("R2_ACCESS_KEY", ""),
    secret_key=os.getenv("R2_SECRET_KEY", ""),
)


def upload(items: list[str]):
    for item in items:
        file_name = os.path.basename(item)
        clean_name = file_name.replace("_windowseat_output", "")
        new_path = os.path.join("ai", clean_name)
        print (f"Uploading {item} to {new_path} in bucket {BUCKET_NAME}")        

        client.fput_object(
            bucket_name=BUCKET_NAME, object_name=new_path, file_path=item
        )


def download(items: list[str]):
    output_dir = os.path.join(CUR_DIR, "example_images")
    for item in items:
        print (f"Downloading {item} from bucket {BUCKET_NAME} to {output_dir}")
        client.fget_object(
            bucket_name=BUCKET_NAME,
            object_name=f"{item}/{item}.jpg",
            file_path=os.path.join(output_dir, f"{item}.jpg"),
        )


def main():
    parser = argparse.ArgumentParser(prog="sync", usage="%(prog)s [options]")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("upload", help="Upload files")

    download_parser = subparsers.add_parser("download", help="Download files")
    download_group = download_parser.add_mutually_exclusive_group(required=True)
    download_group.add_argument("--file", help="Download specific file")
    download_group.add_argument("--imageid", help="Download by image ID")

    args = parser.parse_args()

    if args.command == "upload":
        output = os.path.join(CUR_DIR, "outputs") 
        files_to_upload = [os.path.join(output,f) for f in os.listdir(output) if f.endswith("_windowseat_output.jpg")]
        upload(files_to_upload)
    elif args.command == "download":
        if args.file:
            with open (args.file, "r") as f:
                image_ids = [line.strip() for line in f if line.strip()]
            download(image_ids)
        elif args.imageid:
            download([args.imageid])
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
