from astrbot.api.event import filter, AstrMessageEvent, MessageEventResult
from astrbot.api.star import Context, Star, register
import os
import asyncio
import time


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
        current_dir = os.path.dirname(__file__)
        example_path = os.path.join(current_dir, "vv_pic.py")


        if not os.path.exists(example_path):
            yield event.plain_result("vv_pic doesn't exist！")
            return

        command = f'python "{example_path}" "{query}"'
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            output = stdout.decode('utf-8', errors='ignore') if stdout else ""
            error = stderr.decode('utf-8', errors='ignore') if stderr else ""

            if output:
                # time.sleep(1)
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