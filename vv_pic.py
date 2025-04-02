import requests
import os
import json
from datetime import datetime
import re
import struct
import logging
from typing import Optional, Tuple, List, Dict, Any  # 常用类型注解
# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreviewExtractor:
    """基于二进制索引的高效图片下载器"""

    def __init__(self, base_url: str = "https://vv.noxylva.org"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Accept-Encoding": "identity"  # 强制禁用压缩
        })
        self.timeout = 10  # 适当增加超时时间

    def _fetch_index(self, group_index: int) -> bytes:
        try:
            index_url = f"{self.base_url}/{group_index}.index"
            # 强制禁用所有自动解压
            response = self.session.get(
                index_url,
                timeout=self.timeout,
                headers={"Accept-Encoding": "identity"},  # 关键！
                stream=True
            )
            response.raise_for_status()

            raw_data = response.content

            # 先处理 aws-chunked 分块
            if "aws-chunked" in response.headers.get("content-encoding", ""):
                raw_data = self._decode_aws_chunked(raw_data)

            # 再处理 gzip 压缩
            if "gzip" in response.headers.get("content-encoding", ""):
                import zlib
                try:
                    # 尝试自动检测 gzip/zlib 头
                    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 32)
                    return decompressor.decompress(raw_data)
                except zlib.error:
                    # 如果失败，尝试原始数据（可能已解压）
                    return raw_data

            return raw_data

        except Exception as e:
            logger.error(f"索引下载失败: {group_index}.index - {str(e)}")
            raise

    def _decode_aws_chunked(self, data: bytes) -> bytes:
        """解析 AWS 分块编码"""
        decoded = bytearray()
        pos = 0

        while pos < len(data):
            # 查找分块大小行
            chunk_end = data.find(b"\r\n", pos)
            if chunk_end == -1:
                break

            chunk_size_line = data[pos:chunk_end]
            try:
                chunk_size = int(chunk_size_line, 16)
            except ValueError:
                break

            if chunk_size == 0:
                break  # 结束标志

            # 定位分块数据起始位置
            chunk_start = chunk_end + 2
            chunk_data_end = chunk_start + chunk_size

            if chunk_data_end > len(data):
                break

            decoded.extend(data[chunk_start:chunk_data_end])
            pos = chunk_data_end + 2  # 跳过结尾的 \r\n

        return bytes(decoded)
    def _parse_index(self, index_data: bytes, folder_id: int, frame_num: int) -> Optional[Tuple[int, int]]:
        """解析二进制索引定位图片位置"""
        try:
            # 解析头部信息
            grid_w, grid_h, folder_count = struct.unpack("<III", index_data[:12])
            offset = 12 + folder_count * 4
            file_count = struct.unpack("<I", index_data[offset:offset + 4])[0]
            offset += 4

            # 二分查找定位记录
            left, right = 0, file_count - 1
            while left <= right:
                mid = (left + right) // 2
                record_offset = offset + mid * 16
                curr_folder, curr_frame, curr_offset = struct.unpack("<IIQ",
                                                                     index_data[record_offset:record_offset + 16])

                if curr_folder == folder_id and curr_frame == frame_num:
                    end_offset = struct.unpack("<Q", index_data[record_offset + 24:record_offset + 32])[
                        0] if mid < file_count - 1 else None
                    return (curr_offset, end_offset)
                elif curr_folder < folder_id or (curr_folder == folder_id and curr_frame < frame_num):
                    left = mid + 1
                else:
                    right = mid - 1
            return None
        except Exception as e:
            logger.error(f"索引解析失败: {str(e)}")
            return None

    def extract_frame(self, folder_id: int, frame_num: int) -> Optional[bytes]:
        """主下载方法"""
        try:
            group_index = (folder_id - 1) // 10
            index_data = self._fetch_index(group_index)
            offset_info = self._parse_index(index_data, folder_id, frame_num)

            if not offset_info:
                return None

            start_offset, end_offset = offset_info

            # 构造范围请求
            headers = {"Range": f"bytes={start_offset}-{end_offset - 1}"} if end_offset else {}
            image_url = f"{self.base_url}/{group_index}.webp"

            response = self.session.get(image_url, headers=headers, timeout=self.timeout)
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"下载失败 P{folder_id}-{frame_num}s: {str(e)}")
            return None


def parse_timestamp(timestamp: str) -> int:
    """时间格式转换 (24m26s -> 1466)"""
    match = re.match(r'(\d+)m(\d+)s', timestamp)
    return int(match.group(1)) * 60 + int(match.group(2)) if match else 0


def download_images(query: str, min_ratio: int = 50, min_similarity: float = 0.5, max_results: int = 1):
    """主下载流程"""
    # 初始化下载器
    extractor = PreviewExtractor()

    # 创建保存目录
    # save_dir = f"downloads_{query}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    # os.makedirs(save_dir, exist_ok=True)

    # 保存到根目录
    save_dir = os.path.dirname(__file__)

    try:
        # 获取搜索结果
        response = requests.get(
            "https://vvapi.cicada000.work/search",
            params={
                "query": query,
                "min_ratio": min_ratio,
                "min_similarity": min_similarity,
                "max_results": max_results
            },
            timeout=30
        )
        response.raise_for_status()

        # 解析结果
        results = []
        for line in response.text.splitlines():
            if line.strip():
                try:
                    item = json.loads(line)
                    # 添加对必要字段的检查
                    if all(key in item for key in ['filename', 'timestamp', 'similarity']):
                        results.append(item)
                except json.JSONDecodeError:
                    continue

        if not results:
            print("none", flush=True)
            return

        # 开始下载
        success_count = 0
        for idx, item in enumerate(results, 1):
            try:
                # 提取参数
                folder_id = int(re.search(r'\[P(\d+)\]', item['filename']).group(1))
                frame_num = parse_timestamp(item['timestamp'])

                # 获取图片数据
                image_data = extractor.extract_frame(folder_id, frame_num)
                if not image_data:
                    continue

                # 保存文件
                filename = f"P{folder_id:03d}_{frame_num}s_{item['similarity']:.2f}.webp"
                with open(os.path.join(save_dir, filename), 'wb') as f:
                    f.write(image_data)
                print(filename)
                # logger.info(f"({idx}/{len(results)}) 下载成功: {filename}")
                success_count += 1

            except Exception as e:
                logger.error(f"failed [{idx}]: {str(e)}")

        # 输出统计
        # logger.info(f"\n下载完成！成功率: {success_count}/{len(results)}")
        # logger.info(f"文件保存至: {os.path.abspath(save_dir)}")

    except Exception as e:
        logger.error(f"流程异常: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VV视频帧下载工具")
    parser.add_argument("query", type=str, help="搜索关键词（示例：'测试'）")
    parser.add_argument("--min_ratio", type=int, default=50,
                        help="最小文本匹配度（0-100，默认50）")
    parser.add_argument("--min_similarity", type=float, default=0.5,
                        help="最小人脸相似度（0.0-1.0，默认0.5）")
    parser.add_argument("--max_results", type=int, default=1,
                        help="最大返回结果数（默认1）")

    args = parser.parse_args()

    download_images(
        query=args.query,
        min_ratio=args.min_ratio,
        min_similarity=args.min_similarity,
        max_results=args.max_results
    )