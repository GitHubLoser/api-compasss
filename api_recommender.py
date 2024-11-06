import pinecone
from pinecone import Pinecone
from zhipuai import ZhipuAI
from typing import List, Dict

class APIRecommender:
    def __init__(self, zhipuai_key: str, pinecone_key: str):
        # 初始化智谱AI客户端
        self.client = ZhipuAI(api_key=zhipuai_key)
        
        # 初始化Pinecone（使用新的初始化方式）
        pc = Pinecone(api_key=pinecone_key)
        self.index = pc.Index('api-recommendations')

    def get_embedding(self, text: str) -> List[float]:
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

    def get_api_recommendations(self, user_query: str, top_k: int = 5) -> List[Dict]:
        """根据用户查询返回推荐的API列表"""
        query_vector = self.get_embedding(user_query)
        
        if not query_vector:
            return []
        
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            recommendations = []
            for match in results.matches:
                # 创建基本推荐信息
                recommendation = {
                    'score': match.score,
                    'api_name': match.metadata['api_name'],
                    'description': match.metadata['description']
                }
                # 只有当metadata中存在endpoint时才添加
                if 'endpoint' in match.metadata:
                    recommendation['endpoint'] = match.metadata['endpoint']
                
                recommendations.append(recommendation)
            
            return recommendations
            
        except Exception as e:
            print(f"搜索API推荐时出错: {e}")
            return []

    def stream_chat_response(self, user_query: str, recommendations: List[Dict]):
        """生成流式聊天响应"""
        # 构建系统提示词
        system_prompt = """你是一个专业的API推荐助手。根据用户的需求和提供的API列表，
        你需要解释为什么这些API适合用户的需求，并提供使用建议。请确保回答专业、准确、易懂。"""
        
        # 构建API推荐信息
        api_info = "\n\n推荐的API列表：\n"
        for i, rec in enumerate(recommendations, 1):
            api_info += f"{i}. {rec['api_name']} (相似度: {rec['score']:.2f})\n"
            api_info += f"   描述: {rec['description']}\n"
            if 'endpoint' in rec and rec['endpoint']:  # 只有当endpoint存在且非空时才显示
                api_info += f"   接口: {rec['endpoint']}\n"

        # 创建聊天完成请求
        try:
            response = self.client.chat.completions.create(
                model="glm-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"用户需求: {user_query}\n{api_info}"}
                ],
                stream=True
            )
            
            return response
            
        except Exception as e:
            print(f"生成聊天响应时出错: {e}")
            return None

def main():
    # 配置参数
    ZHIPUAI_KEY = "02dc99f44be91fc5e6345ee2e90b6a27.S3AlhwR7tFNpnlcK"
    PINECONE_KEY = "pcsk_7KKVnM_TyrTke9bYMbGr3z4PTPXJwmThqQD7SDcUw3VM4HmnFoxyqBfgCfpjweDiFX9yZC"
    
    # 创建推荐器实例
    recommender = APIRecommender(ZHIPUAI_KEY, PINECONE_KEY)
    
    while True:
        # 获取用户输入
        user_query = input("\n请输入您的需求（输入 'quit' 退出）: ")
        
        if user_query.lower() == 'quit':
            break
        
        # 获取API推荐
        recommendations = recommender.get_api_recommendations(user_query)
        
        # 生成并输出流式响应
        response = recommender.stream_chat_response(user_query, recommendations)
        if response:
            for chunk in response:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end='')

if __name__ == "__main__":
    main() 