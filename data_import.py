import pandas as pd
from zhipuai import ZhipuAI
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
import time

class APIDataImporter:
    def __init__(self, zhipuai_key: str, pinecone_key: str):
        # 初始化智谱AI客户端
        self.client = ZhipuAI(api_key=zhipuai_key)
        
        # 初始化Pinecone
        pc = Pinecone(api_key=pinecone_key)
        
        try:
            # 直接获取索引，如果不存在会抛出异常
            self.index = pc.Index('api-recommendations')
        except Exception as e:
            print(f"获取索引失败，尝试创建新索引: {e}")
            # 创建新索引
            pc.create_index(
                name='api-recommendations',
                dimension=1024,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # 使用 AWS 的 us-east-1 区域
                )
            )
            time.sleep(5)  # 等待索引创建完成
            self.index = pc.Index('api-recommendations')

    def get_embedding(self, text: str) -> list:
        """获取文本的向量嵌入"""
        try:
            response = self.client.embeddings.create(
                model="embedding-2",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取嵌入向量时出错: {e}")
            return None

    def import_from_excel(self, excel_path: str, batch_size: int = 10):
        """从Excel文件导入API数据到Pinecone"""
        try:
            # 读取Excel文件
            df = pd.read_excel(excel_path)
            print(f"成功读取Excel文件，共有{len(df)}条记录")

            # 确保必填字段存在
            required_fields = ['api_name', 'description']
            for field in required_fields:
                if field not in df.columns:
                    raise ValueError(f"Excel文件中缺少必填字段: {field}")

            # 如果endpoint列不存在，添加空值
            if 'endpoint' not in df.columns:
                df['endpoint'] = ''

            # 批量处理数据
            for i in tqdm(range(0, len(df), batch_size), desc="导入进度"):
                batch = df.iloc[i:i+batch_size]
                vectors_to_upsert = []
                
                for _, row in batch.iterrows():
                    # 构建用于生成向量的文本（只包含非空字段）
                    text_parts = [
                        f"API名称: {row['api_name']}",
                        f"描述: {row['description']}"
                    ]
                    if row['endpoint']:  # 只有当endpoint存在且非空时才添加
                        text_parts.append(f"接口: {row['endpoint']}")
                    
                    text_to_embed = "\n".join(text_parts)
                    embedding = self.get_embedding(text_to_embed)
                    
                    if embedding:
                        # 构建metadata，只包含非空值
                        metadata = {
                            'api_name': row['api_name'],
                            'description': row['description']
                        }
                        if row['endpoint']:
                            metadata['endpoint'] = row['endpoint']

                        vectors_to_upsert.append({
                            'id': str(row.name),
                            'values': embedding,
                            'metadata': metadata
                        })
                
                # 批量上传向量到Pinecone
                if vectors_to_upsert:
                    self.index.upsert(vectors=vectors_to_upsert)
                    time.sleep(1)
            
            print("数据导入完成！")
            
        except Exception as e:
            print(f"导入数据时出错: {e}")

def main():
    # 配置参数
    ZHIPUAI_KEY = "02dc99f44be91fc5e6345ee2e90b6a27.S3AlhwR7tFNpnlcK"  # 替换为您的智谱AI API密钥
    PINECONE_KEY = "pcsk_7KKVnM_TyrTke9bYMbGr3z4PTPXJwmThqQD7SDcUw3VM4HmnFoxyqBfgCfpjweDiFX9yZC"  # 替换为您的Pinecone API密钥
    EXCEL_PATH = "/Users/liwanli/Documents/破局作业/大作业/api.xlsx"  # 替换为您的Excel文件路径
    
    # 初始化 Pinecone
    pc = Pinecone(api_key=PINECONE_KEY)
    
    # 检查现有索引
    print("检查现有索引...")
    existing_indexes = pc.list_indexes().names()
    print(f"现有索引列表: {existing_indexes}")
    
    # 如果索引已存在，先删除
    if 'api-recommendations' in existing_indexes:
        print("删除现有索引...")
        pc.delete_index('api-recommendations')
        time.sleep(5)
    
    # 创建导入器实例并执行导入
    print("开始创建新索引并导入数据...")
    importer = APIDataImporter(ZHIPUAI_KEY, PINECONE_KEY)
    importer.import_from_excel(EXCEL_PATH)

if __name__ == "__main__":
    main() 