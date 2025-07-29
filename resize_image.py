from PIL import Image
import math
import os
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import logging
import shutil

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resize_image.log'),
        logging.StreamHandler()
    ]
)

def resize_to_1920(image_path, output_path=None):
    """
    将图片的最大边长调整为1920，保持宽高比
    """
    try:
        # 打开图片
        img = Image.open(image_path)
        
        # 获取原始尺寸
        width, height = img.size
        
        # 计算缩放比例
        if width > height:
            # 如果宽度大于高度，以宽度为基准
            scale = 3840 / width
            new_width = 3840
            new_height = int(height * scale)
        else:
            # 如果高度大于或等于宽度，以高度为基准
            scale = 3840 / height
            new_height = 3840
            new_width = int(width * scale)
        
        # 调整图片大小
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # 设置输出路径
        if output_path is None:
            output_path = image_path
            
        # 保存图片
        resized_img.save(output_path)
        
        # 关闭图片
        img.close()
        resized_img.close()
        
        return True, f"Successfully processed {image_path}"
    except Exception as e:
        return False, f"Error processing {image_path}: {str(e)}"

def process_pdf_task(pdf_id):
    """
    处理单个PDF的所有图片的任务函数
    """
    try:
        input_dir = f'./data/images_15/{pdf_id}'
        output_dir = f'./data/images_15_3840/{pdf_id}'  # 修改输出目录名
        os.makedirs(output_dir, exist_ok=True)
        
        # 获取所有图片文件
        image_files = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        
        # 按页码排序
        image_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        # 串行处理每个图片
        results = []
        for image_path in tqdm(image_files, desc=f"Processing {pdf_id}", leave=False):
            # 构建输出路径
            rel_path = os.path.relpath(image_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 调整分辨率
            result = resize_to_1920(image_path, output_path)
            results.append(result)
        
        # 统计处理结果
        success = sum(1 for status, _ in results if status)
        total = len(results)
        
        return True, f"Processed PDF {pdf_id}: {success}/{total} images successful"
    except Exception as e:
        return False, f"Error processing PDF {pdf_id}: {str(e)}"

def clean_output_directory():
    """
    删除 images_1920 目录及其所有内容
    """
    output_dir = './data/images_15_3840'  # 修改输出目录名
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            logging.info(f"Successfully removed directory: {output_dir}")
    except Exception as e:
        logging.error(f"Error removing directory {output_dir}: {str(e)}")

def main():
    try:
        # 清理输出目录
        clean_output_directory()
        
        # 获取所有PDF ID
        pdf_ids = os.listdir('./data/images_15')
        
        # 设置进程数
        num_processes = 32
        
        # 使用进程池处理PDF
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(
                pool.imap(process_pdf_task, pdf_ids),
                total=len(pdf_ids),
                desc="Processing PDFs"
            ))
        
        # 输出处理结果
        for status, message in results:
            if status:
                logging.info(message)
            else:
                logging.error(message)
                
    except Exception as e:
        logging.error(f"Error in main process: {str(e)}")

if __name__ == "__main__":
    main()
