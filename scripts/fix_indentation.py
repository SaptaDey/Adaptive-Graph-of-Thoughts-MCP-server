#!/usr/bin/env python3
"""
fix_indentation.py

Refactors Python files to replace tabs with spaces, remove trailing whitespace,
and provide optional dry-run and multi-directory support.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(
        description='Fix indentation: replace tabs with spaces and strip trailing whitespace.'
    )
    parser.add_argument(
        'dirs', nargs='*', default=['src'],
        help='Directories to scan (default: src)'
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Report files that would be modified without writing changes.'
    )
    parser.add_argument(
        '--jobs', type=int, default=1,
        help='Number of parallel worker processes (default: 1).'
    )
    parser.add_argument(
        '--log-level', default='INFO',
        choices=['DEBUG','INFO','WARNING','ERROR'],
        help='Set logging level (default: INFO).'
    )
    return parser.parse_args()

def fix_indentation_issues(filepath: Path, check_only: bool=False) -> bool:
    try:
        text = filepath.read_text(encoding='utf-8')
        cleaned_lines = []
        changed = False
        for line in text.splitlines(keepends=True):
            new_line = line.replace('\t', '    ').rstrip() + '\n'
            if new_line != line:
                changed = True
            cleaned_lines.append(new_line)
        if changed:
            if not check_only:
                filepath.write_text(''.join(cleaned_lines), encoding='utf-8')
            logging.info(
                '%s %s', 'Would modify' if check_only else 'Modified', filepath
            )
        return changed
    except Exception as e:
        logging.error('Error processing %s: %s', filepath, e)
        return False

def collect_python_files(directories):
    for d in directories:
        base = Path(d)
        if not base.exists():
            logging.warning('Directory not found: %s', base)
            continue
        for filepath in base.rglob('*.py'):
            yield filepath

def main():
    args = parse_args()
    logging.basicConfig(level=args.log_level, format='%(levelname)s: %(message)s')
    files = list(collect_python_files(args.dirs))
    results = []
    with Pool(args.jobs) as pool:
        for changed in tqdm(pool.imap_unordered(
            lambda f: fix_indentation_issues(f, args.check), files
        ), total=len(files), desc='Fixing indentation'):
            results.append(changed)
    total_modified = sum(results)
    logging.info('Total files %s: %d/%d',
                 'to change' if args.check else 'modified',
                 total_modified, len(files))

if __name__ == '__main__':
    main()
    logging.basicConfig(level=args.log_level, format='%(levelname)s: %(message)s')
    files = list(collect_python_files(args.dirs))
    results = []
    with Pool(args.jobs) as pool:
        for changed in tqdm(pool.imap_unordered(
            lambda f: fix_indentation_issues(f, args.check), files
        ), total=len(files), desc='Fixing indentation'):
            results.append(changed)
    total_modified = sum(results)
    logging.info('Total files %s: %d/%d',
                 'to change' if args.check else 'modified',
                 total_modified, len(files))

if __name__ == '__main__':
    main()