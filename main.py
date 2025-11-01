from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
import os
import asyncio
import time
import re  # 使用正则表达式进行更灵活的字符验证


@register("vv_pic", "Lonelysky", "", "", "")
class MyPlugin(Star):
    def __init__(self, context: Context):
        super().__init__(context)

    # @filter.command("hello")
    # async def helloworld(self, event: AstrMessageEvent):
    #     user_name = event.get_sender_name()
    #     yield event.plain_result(f"Hello, {user_name}!")

    @filter.command("vvhelp")
    async def help(self, event: AstrMessageEvent):
        yield event.plain_result("输入'/vv 关键词' 进行检索")

    @filter.command("vv")  # 直接注册为指令
    async def execute_script(self, event: AstrMessageEvent, query: str):
        # 增加安全审查，对用户输入进行验证，允许字母、数字、空格、中文和一些常见符号
        # 使用正则表达式匹配允许的字符
        # 这里我们主要使用基本中文字符范围，已经覆盖了绝大部分常用汉字
        allowed_pattern = re.compile(r'^[\w\s\u4e00-\u9fff\u3400-\u4dbf\-_\.]+$')

        if not allowed_pattern.match(query):
            yield event.plain_result("关键词包含非法字符，只允许字母、数字、空格、中文、-、_和.")
            return

        current_dir = os.path.dirname(__file__)
        example_path = os.path.join(current_dir, "vv_pic.py")

        if not os.path.exists(example_path):
            yield event.plain_result("vv_pic doesn't exist！")
            return

        try:
            # 使用create_subprocess_exec而不是create_subprocess_shell
            # 这样可以避免shell解释特殊字符
            proc = await asyncio.create_subprocess_exec(
                "python", example_path, query,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            output = stdout.decode('utf-8', errors='ignore').strip() if stdout else ""
            error = stderr.decode('utf-8', errors='ignore') if stderr else ""

            if output:
                if output == "none":
                    yield event.plain_result(f"未找到与'{query}'相匹配的结果\n建议尝试使用更简短的关键词")
                else:
                    yield event.image_result(os.path.join(current_dir, output).strip())
            if error:
                yield event.plain_result(f"错误：\n{error}")

            for filename in os.listdir(current_dir):
                if filename.endswith(".webp"):
                    file_path = os.path.join(current_dir, filename)
                    try:
                        os.remove(file_path)
                        print(f"已删除文件: {file_path}")
                    except Exception as e:
                        print(f"删除文件失败: {file_path}, 错误: {e}")
        except Exception as e:
            yield event.plain_result(f"执行失败：{str(e)}")

    async def terminate(self):
        pass