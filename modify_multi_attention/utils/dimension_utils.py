"""鍔ㄦ€佺淮搴﹀弬鏁板伐鍏锋ā鍧?
璇ユā鍧楁彁渚涗简浠庨厤缃枃浠跺姩鎬佽鍙栧拰璁剧疆鏁版嵁缁村害鍙傛暟鐨勫伐鍏峰嚱鏁帮紝
閬垮厤鍦ㄤ唬鐮佷腑纭紪鐮佺淮搴﹀€硷紝鎻愰珮浠ｇ爜鐨勭伒娲绘€у拰鍙淮鎶ゆ€с€?"""

import yaml
from typing import Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

def get_data_dimensions_from_config(config: Dict[str, Any]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    浠庨厤缃瓧鍏镐腑鑾峰彇杈撳叆鍜岃緭鍑烘暟鎹淮搴?    
    Args:
        config: 閰嶇疆瀛楀吀
        
    Returns:
        Tuple[Tuple[int, int], Tuple[int, int]]: ((input_h, input_w), (output_h, output_w))
        
    Raises:
        KeyError: 褰撻厤缃腑缂哄皯蹇呰鐨勭淮搴﹀弬鏁版椂
        ValueError: 褰撶淮搴﹀弬鏁版牸寮忎笉姝ｇ‘鏃?    """
    try:
        pdebench_config = config["data"]["pdebench"]
        
        # 鑾峰彇杈撳叆缁村害
        input_size = pdebench_config["input_size"]
        if not isinstance(input_size, list) or len(input_size) != 2:
            raise ValueError(f"input_size蹇呴』鏄暱搴︿负2鐨勫垪琛紝褰撳墠鍊? {input_size}")
        input_h, input_w = int(input_size[0]), int(input_size[1])
        
        # 鑾峰彇杈撳嚭缁村害
        output_size = pdebench_config["output_size"]
        if not isinstance(output_size, list) or len(output_size) != 2:
            raise ValueError(f"output_size蹇呴』鏄暱搴︿负2鐨勫垪琛紝褰撳墠鍊? {output_size}")
        output_h, output_w = int(output_size[0]), int(output_size[1])
        
        logger.info(f"浠庨厤缃腑璇诲彇鏁版嵁缁村害: 杈撳叆({input_h}x{input_w}), 杈撳嚭({output_h}x{output_w})")
        
        return (input_h, input_w), (output_h, output_w)
        
    except KeyError as e:
        raise KeyError(f"閰嶇疆涓己灏戝繀瑕佺殑缁村害鍙傛暟: {e}")
    except (ValueError, TypeError) as e:
        raise ValueError(f"缁村害鍙傛暟鏍煎紡閿欒: {e}")

def get_flattened_dimensions_from_config(config: Dict[str, Any]) -> Tuple[int, int]:
    """
    浠庨厤缃瓧鍏镐腑鑾峰彇灞曞钩鍚庣殑杈撳叆鍜岃緭鍑虹淮搴?    
    Args:
        config: 閰嶇疆瀛楀吀
        
    Returns:
        Tuple[int, int]: (input_dim, output_dim) 灞曞钩鍚庣殑缁村害
    """
    (input_h, input_w), (output_h, output_w) = get_data_dimensions_from_config(config)
    
    input_dim = input_h * input_w
    output_dim = output_h * output_w
    
    logger.info(f"灞曞钩鍚庣淮搴? 杈撳叆({input_dim}), 杈撳嚭({output_dim})")
    
    return input_dim, output_dim

def validate_model_dimensions(config: Dict[str, Any]) -> bool:
    """
    楠岃瘉妯″瀷閰嶇疆涓殑缁村害鍙傛暟鏄惁涓庢暟鎹淮搴︿竴鑷?    
    Args:
        config: 閰嶇疆瀛楀吀
        
    Returns:
        bool: 楠岃瘉鏄惁閫氳繃
    """
    try:
        # 浠庢暟鎹厤缃幏鍙栧疄闄呯淮搴?        actual_input_dim, actual_output_dim = get_flattened_dimensions_from_config(config)
        
        # 浠庢ā鍨嬮厤缃幏鍙栬缃殑缁村害
        model_config = config["model"]
        configured_input_dim = model_config["input_dim"]
        configured_output_dim = model_config["output_dim"]
        
        # 楠岃瘉杈撳叆缁村害
        if actual_input_dim != configured_input_dim:
            logger.warning(
                f"杈撳叆缁村害涓嶅尮閰? 鏁版嵁瀹為檯缁村害({actual_input_dim}) != 妯″瀷閰嶇疆缁村害({configured_input_dim})"
            )
            return False
            
        # 楠岃瘉杈撳嚭缁村害
        if actual_output_dim != configured_output_dim:
            logger.warning(
                f"杈撳嚭缁村害涓嶅尮閰? 鏁版嵁瀹為檯缁村害({actual_output_dim}) != 妯″瀷閰嶇疆缁村害({configured_output_dim})"
            )
            return False
            
        logger.info("妯″瀷缁村害閰嶇疆楠岃瘉閫氳繃")
        return True
        
    except Exception as e:
        logger.error(f"缁村害楠岃瘉澶辫触: {e}")
        return False

def auto_update_model_dimensions(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    鏍规嵁鏁版嵁缁村害鑷姩鏇存柊妯″瀷閰嶇疆涓殑缁村害鍙傛暟
    
    Args:
        config: 閰嶇疆瀛楀吀
        
    Returns:
        Dict[str, Any]: 鏇存柊鍚庣殑閰嶇疆瀛楀吀
    """
    try:
        # 鑾峰彇瀹為檯鏁版嵁缁村害
        actual_input_dim, actual_output_dim = get_flattened_dimensions_from_config(config)
        
        # 鏇存柊妯″瀷閰嶇疆
        config["model"]["input_dim"] = actual_input_dim
        config["model"]["output_dim"] = actual_output_dim
        
        logger.info(f"宸茶嚜鍔ㄦ洿鏂版ā鍨嬬淮搴﹂厤缃? input_dim={actual_input_dim}, output_dim={actual_output_dim}")
        
        return config
        
    except Exception as e:
        logger.error(f"鑷姩鏇存柊妯″瀷缁村害澶辫触: {e}")
        return config

def load_config_with_dimension_validation(config_path: str) -> Dict[str, Any]:
    """
    鍔犺浇閰嶇疆鏂囦欢骞堕獙璇佺淮搴﹀弬鏁?    
    Args:
        config_path: 閰嶇疆鏂囦欢璺緞
        
    Returns:
        Dict[str, Any]: 楠岃瘉鍚庣殑閰嶇疆瀛楀吀
        
    Raises:
        FileNotFoundError: 閰嶇疆鏂囦欢涓嶅瓨鍦?        yaml.YAMLError: YAML鏍煎紡閿欒
        ValueError: 缁村害鍙傛暟楠岃瘉澶辫触
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 楠岃瘉缁村害鍙傛暟
        if not validate_model_dimensions(config):
            logger.warning("缁村害楠岃瘉澶辫触锛屽皾璇曡嚜鍔ㄤ慨姝?..")
            config = auto_update_model_dimensions(config)
            
        return config
        
    except FileNotFoundError:
        raise FileNotFoundError(f"閰嶇疆鏂囦欢涓嶅瓨鍦? {config_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML鏍煎紡閿欒: {e}")

def create_dimension_summary(config: Dict[str, Any]) -> str:
    """
    鍒涘缓缁村害鍙傛暟鎽樿淇℃伅
    
    Args:
        config: 閰嶇疆瀛楀吀
        
    Returns:
        str: 缁村害鍙傛暟鎽樿
    """
    try:
        (input_h, input_w), (output_h, output_w) = get_data_dimensions_from_config(config)
        input_dim, output_dim = get_flattened_dimensions_from_config(config)
        
        summary = f"""
=== 鏁版嵁缁村害鍙傛暟鎽樿 ===
杈撳叆鏁版嵁:
  - 绌洪棿缁村害: {input_h} 脳 {input_w}
  - 灞曞钩缁村害: {input_dim}
  
杈撳嚭鏁版嵁:
  - 绌洪棿缁村害: {output_h} 脳 {output_w}
  - 灞曞钩缁村害: {output_dim}
  
妯″瀷閰嶇疆:
  - input_dim: {config['model']['input_dim']}
  - output_dim: {config['model']['output_dim']}
  
维度匹配状态: {'✅ 匹配' if validate_model_dimensions(config) else '❌ 不匹配'}
========================
"""
        return summary
        
    except Exception as e:
        return f"生成维度摘要失败: {e}"

# 绀轰緥鐢ㄦ硶
if __name__ == "__main__":
    # 娴嬭瘯绀轰緥
    sample_config = {
        "data": {
            "pdebench": {
                "input_size": [32, 32],
                "output_size": [128, 128]
            }
        },
        "model": {
            "input_dim": 1024,
            "output_dim": 16384
        }
    }
    
    print(create_dimension_summary(sample_config))
