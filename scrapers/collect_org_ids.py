import json
import logging
import random
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from .constants import BASE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
}


class GracefulInterruptHandler:
    def __init__(self, timeout_minutes=290):
        self.interrupt_received = False
        self.timeout_received = False
        self.start_time = datetime.now()
        self.timeout_minutes = timeout_minutes
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        logging.info(
            "Received interrupt signal, will save progress and exit...")
        self.interrupt_received = True

    def should_stop(self):
        if datetime.now() - self.start_time > timedelta(minutes=self.timeout_minutes):
            if not self.timeout_received:
                logging.info(
                    f"Reached {self.timeout_minutes} minute timeout, will save progress and exit...")
                self.timeout_received = True
            return True
        return self.interrupt_received or self.timeout_received


class ProgressManager:
    def __init__(self, checkpoint_file: str = "scraper_progress.json"):
        self.checkpoint_file = Path(checkpoint_file)
        self.temp_file = Path(f"{checkpoint_file}.tmp")
        self.backup_file = Path(f"{checkpoint_file}.bak")
        self.scraped_urls = set()
        self.all_player_data = []
        self.load_progress()

    def load_progress(self):
        for file in [self.checkpoint_file, self.temp_file, self.backup_file]:
            try:
                if file.exists():
                    with file.open('r') as f:
                        data = json.load(f)
                        self.all_player_data = data.get('player_data', [])
                        self.scraped_urls = {
                            f"{BASE}/players/{item['ncaa_id']}"
                            for item in self.all_player_data
                        }
                        logging.info(
                            f"Loaded progress from {file}: {len(self.scraped_urls)} URLs scraped, {len(self.all_player_data)} players")
                        return
            except Exception as e:
                logging.warning(f"Error loading from {file}: {e}")
                continue

    def save_progress(self, final=False):
        try:
            if final and self.backup_file.exists():
                self.backup_file.unlink()

            if final and self.checkpoint_file.exists():
                import shutil
                shutil.copy2(str(self.checkpoint_file), str(self.backup_file))

            self.temp_file.parent.mkdir(parents=True, exist_ok=True)

            with self.temp_file.open('w') as f:
                json.dump({
                    'player_data': self.all_player_data
                }, f)

            if self.temp_file.exists():
                if self.checkpoint_file.exists():
                    self.checkpoint_file.unlink()

                import shutil
                shutil.move(str(self.temp_file), str(self.checkpoint_file))

            logging.info(
                f"Progress saved: {len(self.scraped_urls)} URLs scraped, {len(self.all_player_data)} players found")

        except Exception as e:
            logging.error(f"Error saving progress: {e}")
            import traceback
            logging.error(traceback.format_exc())


def get_player_ids_from_career_table(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    ids = []
    career_table = soup.find('table', id=lambda x: x and 'career_totals' in x)
    if career_table:
        for link in career_table.find_all('a'):
            href = link.get('href', '')
            if '/players/' in href:
                player_id = href.split('/players/')[-1].strip()
                if player_id:
                    ids.append(player_id)
    return list(set(ids))


def process_players(urls, timeout_minutes: int = 290) -> pd.DataFrame:
    progress = ProgressManager()
    interrupt_handler = GracefulInterruptHandler(timeout_minutes)
    to_scrape = set(urls)

    processed_ncaa_ids = {item['ncaa_id'] for item in progress.all_player_data}

    to_scrape = to_scrape - progress.scraped_urls

    session = requests.Session()
    session.headers.update(headers)

    try:
        with tqdm(total=len(to_scrape)) as pbar:
            while to_scrape and not interrupt_handler.should_stop():
                current_batch = list(to_scrape)[:10]
                to_scrape = to_scrape - set(current_batch)

                for url in current_batch:
                    if interrupt_handler.should_stop():
                        break

                    try:
                        time.sleep(1 + random.uniform(0, 0.5))
                        player_id = url.split('/players/')[-1].strip()

                        if player_id in processed_ncaa_ids:
                            progress.scraped_urls.add(url)
                            pbar.update(1)
                            continue

                        response = session.get(
                            url, headers=headers, timeout=30)

                        if response.status_code == 430:
                            logging.warning(f"Rate limit hit at URL: {url}")
                            progress.save_progress()
                            return pd.DataFrame(progress.all_player_data)

                        player_ids = get_player_ids_from_career_table(
                            response.content)

                        if player_ids:
                            min_id = min(player_ids)
                            unique_id = f'd3d-{min_id}'

                            for ncaa_id in player_ids:
                                if ncaa_id not in processed_ncaa_ids:
                                    progress.all_player_data.append({
                                        'ncaa_id': ncaa_id,
                                        'unique_id': unique_id
                                    })
                                    processed_ncaa_ids.add(ncaa_id)

                                    progress.scraped_urls.add(
                                        f"{BASE}/players/{ncaa_id}")

                                    url_to_remove = f"{BASE}/players/{ncaa_id}"
                                    if url_to_remove in to_scrape:
                                        to_scrape.remove(url_to_remove)

                        progress.scraped_urls.add(url)
                        pbar.update(1)

                        if len(progress.scraped_urls) % 10 == 0:
                            progress.save_progress()

                    except Exception as e:
                        logging.error(f"Error processing {url}: {e}")
                        continue

    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        progress.save_progress(final=True)

    return pd.DataFrame(progress.all_player_data)


def combine_roster_files(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    all_files = list(output_dir.glob("d*_rosters_*.csv"))
    dfs = []

    for file in tqdm(all_files, desc="Reading roster files"):
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")
            continue

    if not dfs:
        raise ValueError("No valid roster files found")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates()
    logging.info(
        f"Combined {len(all_files)} files into DataFrame with {len(combined_df)} rows")
    return combined_df


def main(data_dir: str) -> None:
    data_dir = Path(data_dir)
    output_dir = data_dir / "rosters"
    output_dir.mkdir(exist_ok=True)

    progress_file = data_dir.parent / 'scraper_progress.json'

    try:
        df = combine_roster_files(output_dir)

        scraped_ids = set()
        ncaa_to_unique_id = {}
        if progress_file.exists():
            try:
                with open(progress_file) as file:
                    scraper_progress = json.load(file)['player_data']
                    scraped_df = pd.DataFrame(data=scraper_progress)
                    ncaa_to_unique_id = dict(
                        zip(scraped_df['ncaa_id'], scraped_df['unique_id'], strict=False))
                    scraped_ids = set(ncaa_to_unique_id.keys())
                    logging.info(
                        f"Loaded {len(scraped_ids)} already scraped IDs from progress file")
            except Exception as e:
                logging.error(f"Error loading progress file: {e}")

        need_to_scrape = []
        for pid in df[~df['player_id'].str.contains('d3d-', na=False)]['player_id'].unique():
            if pid and pid not in scraped_ids:
                need_to_scrape.append(pid)

        logging.info(
            f"Found {len(need_to_scrape)} player IDs that need scraping")

        urls = [
            f'{BASE}/players/{player_id}' for player_id in need_to_scrape]

        result_df = process_players(urls)
        logging.info(
            f"Finished processing with {len(result_df)} player records")

        return 0

    except Exception as e:
        logging.error(f"Fatal error in main: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True,
                        help='Root directory containing the data folders')
    args = parser.parse_args()

    sys.exit(main(args.data_dir))
