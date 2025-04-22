from lane_changing_wrapper import LaneChangingWrapper
import time
def main():
    env = LaneChangingWrapper(render_mode="rgb_array",difficulty='easy')
    s, info = env.reset(seed=0) 
    done = False
    start_time = time.time()
    for i in range(1000):
        # 执行简单动作
        action = (0.2,1)
        s_next, r, dw, tr, info = env.step(action) # dw: dead&win; tr: truncated
        done = (dw or tr)
        s = s_next
    # 计算总耗时
    elapsed_time = time.time() - start_time
    print(f"测试完成，总耗时: {elapsed_time:.2f}秒")

def test_reset():
    env = LaneChangingWrapper(render_mode="rgb_array",difficulty='easy')
    start_time = time.time()
    for i in range(100):
        s, info = env.reset(seed=i)
    elapsed_time = time.time() - start_time
    print(f"测试完成，总耗时: {elapsed_time:.2f}秒")
    
if __name__ == "__main__":
    # main()
    test_reset()