import time

# 记录开始时间
# start = time.time()
#
# # 模拟一段耗时的代码
# time.sleep(2)  # 模拟耗时2秒的操作
#
# # 记录结束时间
# end = time.time()
#
# # 计算并打印执行时间
# print('Time elapsed:', end - start, 'seconds')

b = {'a':[1,2,3,4,5]}

a = b['a']

a.remove(1)
print(a)
print(b)
