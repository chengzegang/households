from typing import (
    AsyncIterable,
    Callable,
)
import os
import httpx
from tqdm.auto import tqdm
import aiofile as aiof
import argparse
import asyncio
from urllib.parse import urlparse


async def download(upstream: AsyncIterable, path: str, callback: Callable):
    """
    upstream: AsyncIterable[bytes] an async iterable of bytes from the remote
    path: str the path to write the file to
    callback: Callable[int] a callback that takes the number of bytes written

    The core design of async download is to overlay the time waiting for streaming with writing to disk.

    """
    async with aiof.async_open(path, "wb+") as f:
        async for block in upstream:
            await f.write(block)
            size = len(block)
            await callback(size)


async def create_upstream(
    url: str, chunk_size: int, callback: Callable
) -> AsyncIterable[bytes]:
    """
    url: str the url to download from
    chunk_size: int the size of the chunks to download
    callback: Callable[int] a callback that takes the total size of the file

    This function open a stream from the url and asynchronously generates chunks of bytes

    """
    async with httpx.AsyncClient(follow_redirects=True, verify=False) as client:
        async with client.stream("GET", url, follow_redirects=True) as resp:
            total_size = int(resp.headers["Content-Length"])
            await callback(total_size)
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(chunk_size):
                yield chunk


async def main(url: str, path: str, chunk_size: int = 4096):
    pbar = tqdm(unit="B", unit_scale=True)

    async def count(total_size: int):
        pbar.reset(total=total_size)

    async def update(n: int):
        pbar.update(n)

    upstream = create_upstream(url, chunk_size, count)
    await download(
        upstream,
        path,
        update,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", type=str)
    parser.add_argument("path", type=str)
    parser.add_argument("-s", "--chunk-size", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if os.path.isfile(args.path) and os.path.exists(args.path) and not args.overwrite:
        raise ValueError("File already exists")
    if os.path.isfile(args.path):
        os.remove(args.path)
    if os.path.isdir(args.path):
        a = urlparse(args.url)
        basename = os.path.basename(a.path)
        args.path = os.path.join(args.path, basename)

    asyncio.run(main(args.url, args.path, args.chunk_size))
