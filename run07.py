import re
import os
import datetime
import subprocess
from http import HTTPStatus
import dashscope
from dashscope import Generation
from moviepy.editor import VideoFileClip, concatenate_videoclips



# 通义千问的API-KEY
MY_DASHSCOPE_API_KEY = 'sk-7be92278538543e1bfc3a71ab47c9833'

dashscope.api_key= MY_DASHSCOPE_API_KEY



# 函数定义
def read_text_from_file(text_file_path):
    """
    功能：
        从指定的文本文件读取内容。
    参数:
        text_file_path:文本文件的路径。
    返回:
        文件中的文本内容(str)。
    """
    with open(text_file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def call_with_messages(text):
    """
    功能：
        接受text内容，并将其分段（以\n\n分隔),。
    参数:
        text:文本内容。
    返回:
        response.output.choices[0].message.content：分段好了之后的文本内容(str)
    """
    messages = [
        {'role': 'system', 'content': '你可以将输入的一大段文本分成几个自然段。'},
        {'role': 'user', 'content': text}
        ]
    response = Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key = "sk-xxx",
        #api_key=os.getenv("sk-7be92278538543e1bfc3a71ab47c9833"), 
        model="qwen-turbo",
        messages=messages,
        result_format="message"
    )

    if response.status_code == 200:
        print(response.output.choices[0].message.content)
        return response.output.choices[0].message.content
    else:
        print(f"HTTP返回码：{response.status_code}")
        print(f"错误码：{response.code}")
        print(f"错误信息：{response.message}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

def save_paragraphs_to_files(paragraphs, folder_path):
    """
    将段落保存为txt文件，依次命名为1、2、3等等。
    
    参数:
        paragraphs (List[str]): 需要保存的段落列表。
        folder_path (str): 文件夹目录路径。
    """
    # 确保文件夹存在
    os.makedirs(folder_path, exist_ok=True)
    
    for i, paragraph in enumerate(paragraphs, start=1):
        file_path = os.path.join(folder_path, f'{i}.txt')
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(paragraph)

def create_output_folder(timestamp, base_folder, file_name):
    """
    功能；
        创建以文件名和当前时间结合的文件夹。
    参数：
        timestamp：时间有关的内容
        base_folder：即将要创建新文件夹的最基础的文件夹
        file_name：新文件夹的名字
    返回：
        folder_path:新文件夹的路径
    """
    folder_name = f"{os.path.splitext(file_name)[0]}_{timestamp}"
    folder_path = os.path.join(base_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def save_files(text, folder_path, file_name):
    """
    功能：
        通用型，将text内容保存为txt文件，存储在folder_path的文件夹中
    参数:
        text(str): 需要保存的text内容。
        folder_path (str): 文件夹目录路径。
        file_name: 文件名称
    返回：
        file_path
    """
    # 确保文件夹存在
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name + '.txt')
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)
    return file_path

def create_Zi_folder(base_folder, folder_name):
    """
    功能：
        用于存放SadTalker生成的一系列不同人物的视频的文件夹
    参数：
        base_folder：马上要创建新文件夹的地方
        file_name：文件夹的名称
    返回：
        new_folder_path：新文件夹的路径
    """
    # 创建完整的文件夹路径
    new_folder_path = os.path.join(base_folder, folder_name)

    # 创建文件夹（如果不存在）
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
        print(f'文件夹已创建: {new_folder_path}')
    else:
        print(f'文件夹已存在: {new_folder_path}')
    return new_folder_path

def run_command_GSV(GSV_work_dir, gpt_model_path, sovits_model_path, ref_audio_path, ref_text_path, ref_language, target_text_path, target_language, GSV_output_path):
    """
    功能：用cmd调用GPT-SoVITS
    参数：
        GSV_work_dir：所使用的GPT-SoVITS的路径
        gpt_model_path：gpt_model路径
        sovits_model_path：sovits_model路径
        ref_audio_path：ref_audio路径
        ref_text_path：ref_text路径
        ref_language：参考音频的语言
        target_text_path：目标文本的路径
        target_language_path：目标文本的语言
        GSV_output_path：GSV_output路径
    返回：
        无
    """
    
    cmd = [
        r'F:\AIAnchor\GPT-SoVITS-beta0706\runtime\python.exe',
        r'F:\AIAnchor\GPT-SoVITS-beta0706\GPT_SoVITS\inference_cli.py',
        '--gpt_model', gpt_model_path,
        '--sovits_model', sovits_model_path,
        '--ref_audio', ref_audio_path,
        '--ref_text', ref_text_path,
        '--ref_language', ref_language,
        '--target_text', target_text_path,
        '--target_language', target_language,
        '--output_path', GSV_output_path
    ]

    # 运行GPT-SoVITS命令
    os.chdir(GSV_work_dir)
    subprocess.run(cmd, check=True)

def run_command_SD(SD_work_dir, driven_audio, source_image, SD_output_path, preprocess, enhancer):
    """
    功能：用cmd调用SadTalker
    参数：
        SD_work_dir：所使用的SadTalker的路径
        driven_audio：驱动音频的路径
        source_image：人物图像的路径
        SD_output_path：SD_output路径
        preprocess：
        enhancer：
    返回：
        无
    """
    
    cmd = [
        r'F:\anaconda3\envs\sadtalker03\python.exe',
        r'F:\AIAnchor\SadTalker\inference.py',
        '--driven_audio', driven_audio,
        '--source_image', source_image,
        '--result_dir', SD_output_path,
        '--still',
        '--preprocess', preprocess,
        '--enhancer', enhancer
    ]

    # 运行SadTalker命令
    os.chdir(SD_work_dir)
    subprocess.run(cmd, check=True)

# 获取GPT模型人物(person)路径
def get_gpt_model_path(code):
    return gpt_model_paths.get(code, "GPT模型人物音色(person)路径未找到")

# 获取SoVITS模型人物音色(person)路径
def get_sovits_model_path(code):
    return sovits_model_paths.get(code, "SoVITS模型人物音色(person)路径未找到")

# 获取人物参考音频（语气）(emotion)路径
def get_ref_audio_path(code):
    return ref_audio_paths.get(code, "人物参考音频（语气）(emotion)路径未找到")

# 获取人物参考音频（语气）(emotion)内容路径
def get_ref_text_path(code):
    return ref_text_paths.get(code, "人物参考音频（语气）(emotion)内容路径未找到")

# 获取人物图像路径
def get_source_image(code):
    return source_images.get(code, "人物图像路径未找到")



# 参数
# Paragraphs
base_Paragraphs_output_path = r'F:\AIAnchor\Output\Paragraphs_Output'


# GPT-SoVITS
# GPT-SoVITS目录
GSV_work_dir = r'F:\AIAnchor\GPT-SoVITS-beta0706'
# GPT-SoVITS参数
# 定义“GPT模型人物(person.ckpt)”路径字典
gpt_model_paths = {
    'HuTao': r"F:\AIAnchor\Input\models\HuTao\hutao-e75.ckpt",
    'HuaHuo': r"F:\AIAnchor\Input\models\HuaHuo\huahuo-e100.ckpt",
    'gdg': r"F:\AIAnchor\Input\models\gdg\gdg-e15.ckpt",
    'yq': r"F:\AIAnchor\Input\models\yq\yq-e15.ckpt"
    # 添加其他GPT模型人物(person.ckpt)
}
# 定义“SoVITS模型人物音色(person.pth)”路径字典
sovits_model_paths = {
    'HuTao': r"F:\AIAnchor\Input\models\HuTao\hutao_e60_s3360.pth",
    'HuaHuo': r"F:\AIAnchor\Input\models\HuaHuo\huahuo_e100_s1100.pth",
    'gdg': r"F:\AIAnchor\Input\models\gdg\gdg_e8_s160.pth",
    'yq': r"F:\AIAnchor\Input\models\yq\yq_e8_s184.pth"
    # 添加其他SoVITS模型人物音色(person.pth)路径
}
# 定义“人物参考音频(reference_i.wav)”路径字典
ref_audio_paths = {
    'HuTao_1': r"F:\AIAnchor\Input\models\HuTao\reference_1\参考音频_村里的气氛还不错，我随便转转就有了诗性。.wav",
    'HuTao_2': r"F:\AIAnchor\Input\models\HuTao\reference_2\参考音频_我说白术，你不会看不出来吧？难不成你师父，忘了教你这门功夫？.wav",
    'HuaHuo_1': r"F:\AIAnchor\Input\models\HuaHuo\reference_1\参考音频_哎呀，别板着脸嘛～还一本正经地引经据典，干嘛这么严肃？.wav",
    'gdg_1': r"F:\AIAnchor\Input\models\gdg\reference_1\参考音频_对了知音谈几句，不对知音枉费。.wav",
    'yq_1': r"F:\AIAnchor\Input\models\yq\reference_1\参考音频_就能看得出来，刚才我们在后台一进门的时候这个安保非常非常地严格。.wav"
    # 添加其他人物参考音频(reference.wav)路径
}
# 定义“人物参考音频文本(reference_i.txt)”路径字典
ref_text_paths = {
    'HuTao_1': r"F:\AIAnchor\Input\models\HuTao\reference_1\参考音频内容.txt",
    'HuaHuo_1': r"F:\AIAnchor\Input\models\HuaHuo\reference_1\参考音频内容.txt",
    'gdg_1': r"F:\AIAnchor\Input\models\gdg\reference_1\参考音频内容.txt",
    'yq_1': r"F:\AIAnchor\Input\models\yq\reference_1\参考音频内容.txt"
    # 添加其他人物参考音频文本(reference.txt)内容路径
}
# 其他参数
ref_language = '中文'
target_language = '中文'
base_GSV_output_path = r'F:\AIAnchor\Output\GPT-SoVITS_Output'


# SadTalker
# SadTalker目录
SD_work_dir = r'F:\AIAnchor\SadTalker'
# SadTalker参数
# 定义“人物图像(person_i.jpg)”路径字典
source_images = {
    'HuTao_1_3x4': r"F:\AIAnchor\Input\models\HuTao\HuTao_1_3x4.jpg",
    'HuaHuo_1_3x4': r"F:\AIAnchor\Input\models\HuaHuo\HuaHuo_1_3x4.jpg",
    'gdg_1_3x4': r"F:\AIAnchor\Input\models\gdg\gdg_1_3x4.jpg",
    'yq_1_3x4': r"F:\AIAnchor\Input\models\yq\yq_1_3x4.jpg"
    # 添加其他人物图像路径
}
# 其他参数
base_SD_output_path = r'F:\AIAnchor\Output\SadTalker_Output'
preprocess = 'full'
enhancer = 'gfpgan'


# VideoMerging
base_VideoMerging_output_path = r'F:\AIAnchor\Output\VideoMerging_Output'




if __name__ == '__main__':
    # 记录时间
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # 确定要处理的文稿的地址
    text_file_path = r"F:\AIAnchor\Input\Paragraphs_Input\advantages.txt"

    # 获取文稿txt文件的名称(带后缀)
    text_file_name = os.path.basename(text_file_path)
    # 创建文件名和当前时间结合的文件夹
    Paragraphs_output_path = create_output_folder(timestamp, base_Paragraphs_output_path, text_file_name)
    GSV_output_path = create_output_folder(timestamp, base_GSV_output_path, text_file_name)
    SD_output_path = create_output_folder(timestamp, base_SD_output_path, text_file_name)
    VideoMerging_output_path = create_output_folder(timestamp, base_VideoMerging_output_path, text_file_name)
    # 创建一个空的视频列表
    video_files = []

    #Para部分
    # 从文本路径所对应的文本读出，存到text里
    text = read_text_from_file(text_file_path)
    # 将text传入大模型，传出分好段的文本（每段以\n\n分隔）(str)并存在text_para里
    text_para = call_with_messages(text)
    # 将text_para存到文件夹目录为Paragraphs_output_path里的text_para.txt文件中
    save_files(text_para, Paragraphs_output_path, 'text_para')
    # 使用 '\n\n' 分割text_para，存储到paragraphs中
    paragraphs = text_para.split('\n\n')

    #按奇数偶数分配读稿子
    for i, paragraph in enumerate(paragraphs):
        k = i + 1
        print(f"段落 {k}:")
        print(paragraph)
        print()  # 打印空行以分隔段落

        # 奇数时选择的人物、音色、语气、参考音频
        odd_choice = {
                        'gpt_model_path' : get_gpt_model_path('gdg'),
                        'sovits_model_path' : get_sovits_model_path('gdg'),
                        'ref_audio_path' : get_ref_audio_path('gdg_1'),
                        'ref_text_path' : get_ref_text_path('gdg_1'),
                        'source_image' : get_source_image('gdg_1_3x4'),
        }
        even_choice = {
                        'gpt_model_path' : get_gpt_model_path('yq'),
                        'sovits_model_path' : get_sovits_model_path('yq'),
                        'ref_audio_path' : get_ref_audio_path('yq_1'),
                        'ref_text_path' : get_ref_text_path('yq_1'),
                        'source_image' : get_source_image('yq_1_3x4'),
        }
        
        if k % 2 == 1:
            m_choice = odd_choice
        else:
            m_choice = even_choice
        print("现在是",k)
        Zi_Paragraphs_output_path = create_Zi_folder(Paragraphs_output_path, str(k))
        Zi_GSV_output_path = create_Zi_folder(GSV_output_path, str(k))
        Zi_SD_output_path = create_Zi_folder(SD_output_path, str(k))
        # 将paragraph保存到txt文件中
        target_text_path = save_files(paragraph, Zi_Paragraphs_output_path, str(k))
        # GPT-SoVITS开始运行
        run_command_GSV(GSV_work_dir, 
                        m_choice['gpt_model_path'], 
                        m_choice['sovits_model_path'], 
                        m_choice['ref_audio_path'], 
                        m_choice['ref_text_path'], 
                        ref_language, 
                        target_text_path, 
                        target_language, 
                        Zi_GSV_output_path)
        # SadTalker开始运行
        driven_audio = os.path.join(Zi_GSV_output_path,'output.wav')
        run_command_SD(SD_work_dir, driven_audio, m_choice['source_image'], Zi_SD_output_path, preprocess, enhancer)
        video_name = os.path.splitext(os.path.basename(m_choice['source_image']))[0]+'##output_enhanced.mp4'
        video_path = os.path.join(Zi_SD_output_path, video_name)
        # 将生成的视频增加到视频列表
        video_files.append(video_path)

    # 视频合并
    # 加载视频文件
    clips = [] #加载视频用的临时存储
    for video in video_files:
        try:
            target_resolution = (960, 1280)  # 目标分辨率 3:4
            clip = VideoFileClip(video).resize(target_resolution) # 将视频调整为设定的分辨率
            clips.append(clip)
        except Exception as e:
            print(f"Failed to load video {video}: {e}")
    # 合并视频片段，使用 method='compose'
    final_clip = concatenate_videoclips(clips, method='compose')
    VideoMerging_video_path = os.path.join(VideoMerging_output_path, os.path.splitext(text_file_name)[0] + '.mp4')
    # 保存视频
    try:
        final_clip.write_videofile(
            VideoMerging_video_path, 
            codec='libx264', 
            audio_codec='aac', 
            threads=4
        )
    except Exception as e:
        print(f"写入视频文件时发生错误: {e}")
