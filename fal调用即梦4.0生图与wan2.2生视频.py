import os
import gradio as gr
import fal_client
from PIL import Image
import tempfile
import shutil
import requests
from io import BytesIO
from typing import List, Optional, Tuple
import base64

# 从环境变量获取 API Key
FAL_KEY = os.environ.get('FAL_KEY', '你的api')

def upload_image_to_fal(image_path: str) -> str:
    """上传本地图片到 FAL 服务器并返回 URL"""
    try:
        # 处理中文路径：读取文件内容后用临时英文路径上传
        try:
            image_path.encode('ascii')
            url = fal_client.upload_file(image_path)
        except UnicodeEncodeError:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    temp_path = tmp.name
                shutil.copy2(image_path, temp_path)
                url = fal_client.upload_file(temp_path)
                return url
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception:
                        pass
        return url
    except Exception as e:
        raise Exception(f"图片上传失败: {str(e)}")

def download_image_from_url(url: str) -> str:
    """从URL下载图片并保存为临时文件"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        img = Image.open(BytesIO(response.content))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            img.save(tmp.name, 'PNG')
            return tmp.name
    except Exception as e:
        raise Exception(f"下载图片失败: {str(e)}")

def process_image_edit(
    images: List,
    image_urls_text: str,
    prompt: str,
    image_size: str,
    custom_width: int,
    custom_height: int,
    num_images: int,
    max_images: int,
    seed: Optional[int],
    enable_safety: bool,
    progress=gr.Progress()
) -> Tuple[List, str]:
    """处理图片编辑请求 (SeedDream 4)"""
    
    if not FAL_KEY:
        return None, "❌ 错误: 未设置 FAL_KEY 环境变量\n请运行: export FAL_KEY='your_api_key'"
    
    os.environ['FAL_KEY'] = FAL_KEY
    
    # 处理图片输入
    all_image_paths = []
    
    # 处理上传的文件
    if images and len(images) > 0:
        all_image_paths.extend(images)
    
    # 处理粘贴的URL
    if image_urls_text and image_urls_text.strip():
        urls = [url.strip() for url in image_urls_text.strip().split('\n') if url.strip()]
        for url in urls:
            try:
                progress(0.05, desc=f"下载图片: {url[:50]}...")
                img_path = download_image_from_url(url)
                all_image_paths.append(img_path)
            except Exception as e:
                return None, f"❌ 错误: 无法下载图片 {url}\n{str(e)}"
    
    if len(all_image_paths) == 0:
        return None, "❌ 错误: 请至少上传一张图片或粘贴图片URL"
    
    if not prompt.strip():
        return None, "❌ 错误: 请输入编辑提示词"
    
    if len(all_image_paths) > 10:
        return None, "❌ 错误: 最多只能处理 10 张图片"
    
    try:
        progress(0.1, desc="正在上传图片...")
        
        image_urls = []
        for i, img_path in enumerate(all_image_paths):
            progress((i + 1) / (len(all_image_paths) + 1) * 0.3, desc=f"上传图片 {i+1}/{len(all_image_paths)}...")
            url = upload_image_to_fal(img_path)
            image_urls.append(url)
        
        progress(0.4, desc="构建请求参数...")
        
        arguments = {
            "prompt": prompt,
            "image_urls": image_urls,
            "num_images": num_images,
            "max_images": max_images,
            "enable_safety_checker": enable_safety,
        }
        
        if image_size == "custom":
            arguments["image_size"] = {"width": custom_width, "height": custom_height}
        else:
            arguments["image_size"] = image_size
        
        if seed is not None and seed > 0:
            arguments["seed"] = seed
        
        progress(0.5, desc="发送请求到 API...")
        
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    progress(0.5, desc=f"生成中: {log['message'][:50]}...")
        
        result = fal_client.subscribe(
            "fal-ai/bytedance/seedream/v4/edit",
            arguments=arguments,
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        
        progress(0.9, desc="处理结果...")
        
        if 'images' not in result or len(result['images']) == 0:
            return None, "❌ 错误: 未生成图片"
        
        output_images = [img_data['url'] for img_data in result['images']]
        
        progress(1.0, desc="完成!")
        
        status = f"""
✅ 图片编辑成功！

📊 生成信息:
- 输入图片: {len(image_urls)} 张
- 输出图片: {len(output_images)} 张
- 使用种子: {result.get('seed', '随机')}
- 提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}

🔗 图片链接:
"""
        for i, url in enumerate(output_images, 1):
            status += f"{i}. {url}\n"
        
        return output_images, status
        
    except Exception as e:
        return None, f"❌ 生成失败\n\n错误信息: {str(e)}"

def process_image_to_video(
    image: Optional[str],
    image_url_text: str,
    prompt: str,
    num_frames: int,
    fps: int,
    resolution: str,
    aspect_ratio: str,
    num_inference_steps: int,
    seed: Optional[int],
    enable_safety: bool,
    enable_output_safety: bool,
    enable_prompt_expansion: bool,
    guidance_scale: float,
    video_quality: str,
    progress=gr.Progress()
) -> Tuple[Optional[str], str]:
    """处理图片转视频请求 (Wan v2.2)"""
    
    if not FAL_KEY:
        return None, "❌ 错误: 未设置 FAL_KEY 环境变量\n请运行: export FAL_KEY='your_api_key'"
    
    os.environ['FAL_KEY'] = FAL_KEY
    
    # 确定图片来源
    image_path = None
    if image_url_text and image_url_text.strip():
        # 使用粘贴的URL
        try:
            progress(0.05, desc="下载图片...")
            image_path = download_image_from_url(image_url_text.strip())
        except Exception as e:
            return None, f"❌ 错误: 无法下载图片\n{str(e)}"
    elif image:
        # 使用上传的图片
        image_path = image
    else:
        return None, "❌ 错误: 请上传图片或粘贴图片URL"
    
    if not prompt.strip():
        return None, "❌ 错误: 请输入视频描述提示词"
    
    try:
        progress(0.1, desc="上传图片到服务器...")
        image_url = upload_image_to_fal(image_path)
        
        progress(0.2, desc="构建请求参数...")
        
        arguments = {
            "image_url": image_url,
            "prompt": prompt,
            "num_frames": num_frames,
            "frames_per_second": fps,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "num_inference_steps": num_inference_steps,
            "enable_safety_checker": enable_safety,
            "enable_output_safety_checker": enable_output_safety,
            "enable_prompt_expansion": enable_prompt_expansion,
            "guidance_scale": guidance_scale,
            "video_quality": video_quality,
        }
        
        if seed is not None and seed > 0:
            arguments["seed"] = seed
        
        progress(0.3, desc="发送请求到 API...")
        
        def on_queue_update(update):
            if isinstance(update, fal_client.InProgress):
                for log in update.logs:
                    progress(0.5, desc=f"生成视频: {log['message'][:50]}...")
        
        result = fal_client.subscribe(
            "fal-ai/wan/v2.2-a14b/image-to-video",
            arguments=arguments,
            with_logs=True,
            on_queue_update=on_queue_update,
        )
        
        progress(0.95, desc="处理结果...")
        
        if 'video' not in result:
            return None, "❌ 错误: 未生成视频"
        
        video_url = result['video']['url']
        
        progress(1.0, desc="完成!")
        
        status = f"""
✅ 视频生成成功！

📊 生成信息:
- 输入图片: 1 张
- 视频帧数: {num_frames}
- 帧率: {fps} FPS
- 分辨率: {resolution}
- 宽高比: {aspect_ratio}
- 使用种子: {result.get('seed', '随机')}
- 提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}

🔗 视频链接:
{video_url}
"""
        
        return video_url, status
        
    except Exception as e:
        return None, f"❌ 生成失败\n\n错误信息: {str(e)}"

def create_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="AI 图片/视频生成器", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # 🎨 AI 图片/视频生成工具
        
        **使用前请确保已设置环境变量:** `export FAL_KEY="your_api_key"`
        
        支持两种功能：
        1. **图片编辑** - 使用 SeedDream 4 编辑和修改图片
        2. **图片转视频** - 使用 Wan v2.2 将静态图片转换为动态视频
        """)
        
        with gr.Tabs():
            # ========== Tab 1: 图片编辑 ==========
            with gr.TabItem("🖼️ 图片编辑 (SeedDream 4)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 1. 输入图片")
                        
                        with gr.Tabs():
                            with gr.TabItem("上传文件"):
                                edit_images_input = gr.File(
                                    label="上传图片（最多10张）",
                                    file_count="multiple",
                                    file_types=["image"],
                                    type="filepath"
                                )
                            
                            with gr.TabItem("粘贴链接"):
                                edit_urls_input = gr.Textbox(
                                    label="图片URL（每行一个）",
                                    placeholder="https://example.com/image1.jpg\nhttps://example.com/image2.jpg",
                                    lines=5
                                )
                        
                        # 图片预览
                        edit_preview = gr.Gallery(
                            label="图片预览",
                            columns=3,
                            height=200,
                            show_label=True
                        )
                        
                        gr.Markdown("### ✍️ 2. 编辑提示词")
                        edit_prompt_input = gr.Textbox(
                            label="编辑提示词",
                            placeholder="描述你想要的编辑效果...",
                            lines=4
                        )
                        
                        with gr.Accordion("⚙️ 高级设置", open=False):
                            edit_image_size = gr.Radio(
                                choices=["auto", "auto_2K", "auto_4K", "square_hd", "square",
                                        "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9", "custom"],
                                value="auto",
                                label="图片尺寸"
                            )
                            
                            with gr.Row(visible=False) as edit_custom_size:
                                edit_width = gr.Number(label="宽度", value=1280, minimum=512, maximum=4096)
                                edit_height = gr.Number(label="高度", value=720, minimum=512, maximum=4096)
                            
                            with gr.Row():
                                edit_num_images = gr.Slider(1, 10, 1, step=1, label="生成数量")
                                edit_max_images = gr.Slider(1, 10, 1, step=1, label="最大图片数")
                            
                            edit_seed = gr.Number(label="随机种子（可选）", value=None, precision=0)
                            edit_safety = gr.Checkbox(label="启用安全检查", value=True)
                        
                        edit_btn = gr.Button("✨ 开始编辑图片", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🖼️ 生成结果")
                        edit_output = gr.Gallery(label="生成的图片", columns=2, height="auto")
                        edit_status = gr.Textbox(label="状态信息", lines=10, interactive=False)
                
                # 预览功能
                def update_edit_preview(files, urls):
                    previews = []
                    if files:
                        previews.extend(files[:10])
                    if urls and urls.strip():
                        url_list = [u.strip() for u in urls.strip().split('\n') if u.strip()]
                        for url in url_list[:10-len(previews)]:
                            try:
                                previews.append(download_image_from_url(url))
                            except:
                                pass
                    return previews
                
                edit_images_input.change(update_edit_preview, [edit_images_input, edit_urls_input], edit_preview)
                edit_urls_input.change(update_edit_preview, [edit_images_input, edit_urls_input], edit_preview)
                
                edit_image_size.change(
                    lambda x: gr.update(visible=(x == "custom")),
                    edit_image_size,
                    edit_custom_size
                )
                
                edit_btn.click(
                    process_image_edit,
                    [edit_images_input, edit_urls_input, edit_prompt_input, edit_image_size,
                     edit_width, edit_height, edit_num_images, edit_max_images, edit_seed, edit_safety],
                    [edit_output, edit_status]
                )
            
            # ========== Tab 2: 图片转视频 ==========
            with gr.TabItem("🎬 图片转视频 (Wan v2.2)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 1. 输入图片")
                        
                        with gr.Tabs():
                            with gr.TabItem("上传文件"):
                                video_image_input = gr.Image(
                                    label="上传图片",
                                    type="filepath"
                                )
                            
                            with gr.TabItem("粘贴链接"):
                                video_url_input = gr.Textbox(
                                    label="图片URL",
                                    placeholder="https://example.com/image.jpg",
                                    lines=2
                                )
                        
                        gr.Markdown("### ✍️ 2. 视频描述")
                        video_prompt_input = gr.Textbox(
                            label="视频描述提示词",
                            placeholder="描述你想要的视频效果和动作...",
                            lines=4
                        )
                        
                        gr.Markdown("### ⚙️ 3. 视频参数")
                        with gr.Row():
                            video_num_frames = gr.Slider(17, 161, 81, step=1, label="帧数")
                            video_fps = gr.Slider(4, 60, 16, step=1, label="帧率 (FPS)")
                        
                        with gr.Row():
                            video_resolution = gr.Radio(
                                choices=["480p", "580p", "720p"],
                                value="720p",
                                label="分辨率"
                            )
                            video_aspect_ratio = gr.Radio(
                                choices=["auto", "16:9", "9:16", "1:1"],
                                value="auto",
                                label="宽高比"
                            )
                        
                        with gr.Accordion("⚙️ 高级设置", open=False):
                            video_steps = gr.Slider(1, 50, 27, step=1, label="推理步数")
                            video_guidance = gr.Slider(1.0, 10.0, 3.5, step=0.1, label="引导强度")
                            video_quality = gr.Radio(
                                choices=["low", "medium", "high", "maximum"],
                                value="high",
                                label="视频质量"
                            )
                            video_seed = gr.Number(label="随机种子（可选）", value=None, precision=0)
                            
                            with gr.Row():
                                video_safety = gr.Checkbox(label="输入安全检查", value=True)
                                video_output_safety = gr.Checkbox(label="输出安全检查", value=False)
                            
                            video_prompt_expansion = gr.Checkbox(label="启用提示词扩展", value=False)
                        
                        video_btn = gr.Button("🎬 生成视频", variant="primary", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎬 生成结果")
                        video_output = gr.Video(label="生成的视频")
                        video_status = gr.Textbox(label="状态信息", lines=10, interactive=False)
                
                video_btn.click(
                    process_image_to_video,
                    [video_image_input, video_url_input, video_prompt_input,
                     video_num_frames, video_fps, video_resolution, video_aspect_ratio,
                     video_steps, video_seed, video_safety, video_output_safety,
                     video_prompt_expansion, video_guidance, video_quality],
                    [video_output, video_status]
                )
        
        gr.Markdown("""
        ---
        ### 💡 使用提示
        
        **图片编辑:**
        - 支持上传文件或粘贴图片链接
        - 可以同时使用多张图片
        - 提示词用英文效果更好
        
        **图片转视频:**
        - 建议使用高质量、清晰的输入图片
        - 提示词要详细描述期望的动作和场景变化
        - 帧数越多，视频越长，但生成时间也越长
        
        ### 📝 示例提示词
        
        **编辑:** "Change the background to a beach at sunset"
        **视频:** "The camera slowly zooms in while the character turns their head"
        """)
    
    return app

if __name__ == "__main__":
    if not FAL_KEY:
        print("=" * 60)
        print("⚠️  警告: 未检测到 FAL_KEY 环境变量")
        print("=" * 60)
        print("请先设置 API Key:")
        print("  export FAL_KEY='your_api_key_here'")
        print()
        print("界面将启动，但无法生成内容直到设置了正确的 API Key")
        print("=" * 60)
        print()
    else:
        print("✅ 检测到 FAL_KEY 环境变量")
        print("🚀 正在启动 Gradio 界面...")
        print()
    
    app = create_interface()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )