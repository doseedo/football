#!/usr/bin/env python3
"""Download SoccerNet dataset for training."""

import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Download SoccerNet dataset')
    parser.add_argument('--output', '-o', default='data/soccernet', help='Output directory')
    parser.add_argument('--task', '-t', default='tracking', 
                        choices=['tracking', 'calibration', 'reid', 'all'],
                        help='Which task data to download')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    try:
        from SoccerNet.Downloader import SoccerNetDownloader
    except ImportError:
        print("Installing SoccerNet package...")
        os.system('pip install SoccerNet')
        from SoccerNet.Downloader import SoccerNetDownloader
    
    downloader = SoccerNetDownloader(LocalDirectory=args.output)
    
    if args.task in ['tracking', 'all']:
        print("\n=== Downloading Tracking Data ===")
        # Player tracking annotations
        downloader.downloadDataTask(task="tracking", split=["train", "valid", "test", "challenge"])
    
    if args.task in ['calibration', 'all']:
        print("\n=== Downloading Calibration Data ===")
        # Camera calibration / homography data
        downloader.downloadDataTask(task="calibration", split=["train", "valid", "test", "challenge"])
    
    if args.task in ['reid', 'all']:
        print("\n=== Downloading Re-ID Data ===")
        # Player re-identification data
        downloader.downloadDataTask(task="reid", split=["train", "valid", "test", "challenge"])
    
    print(f"\n✓ Download complete! Data saved to: {args.output}")
    print("\nDataset structure:")
    os.system(f'find {args.output} -type d | head -20')

if __name__ == '__main__':
    main()
