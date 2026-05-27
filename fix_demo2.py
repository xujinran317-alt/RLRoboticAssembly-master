# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'E:\LeStoreDownload\dailypython\bimros\RLRoboticAssembly-master\RLRoboticAssembly-master\scripts\demo_auto_assembly.py'

with open(path, 'rb') as f:
    data = bytearray(f.read())

# Check for problematic bytes around key string areas
# We need to find and fix corrupted Chinese characters
# The issue: during PowerShell replacement, some Chinese UTF-8 bytes 
# got converted to ASCII '?' (0x3f)

# Let's first find all occurrences of 0x3f that shouldn't be there
# In Chinese text that uses UTF-8, '?' shouldn't appear in the middle of 
# multi-byte sequences

# Check line 103: pr_info("自动装配演示（比例控制，无需训练）")
# The right parenthesis ）should be ef bc 89
# Likely corrupted to ef bc 3f (which makes it look like ef bc ? )

# Let's search for the specific pattern
patterns_to_fix = [
    # 自动装配演示（比例控制，无需训练?  -> 自动装配演示（比例控制，无需训练）
    (b'\xe8\xae\xad\xe7\xbb\x83\xef\xbc\x3f', b'\xe8\xae\xad\xe7\xbb\x83\xef\xbc\x89'),
    # 可直接使用?  -> 可直接使用）
    (b'\xe4\xbd\xbf\xe7\x94\xa8\xef\xbc\x3f', b'\xe4\xbd\xbf\xe7\x94\xa8\xef\xbc\x89'),
    # 正?=  -> 正值=
    (b'\xe6\xad\xa3\xef\xbc\x3f', b'\xe6\xad\xa3\xe5\x80\xbc'),
    # 阈值? -> 阈值
    (b'\xe9\x98\x88\xef\xbc\x3f', b'\xe9\x98\x88\xe5\x80\xbc'),
    # 步数? -> 步数
    (b'\xe6\xad\xa5\xef\xbc\x3f', b'\xe6\xad\xa5\xe6\x95\xb0'),
    # 增益? -> 增益
    (b'\xe5\xa2\x9e\xe7\x9b\x8a\xef\xbc\x3f', b'\xe5\xa2\x9e\xe7\x9b\x8a'),
    # 周期? -> 周期
    (b'\xe5\x91\xa8\xe6\x9c\x9f\xef\xbc\x3f', b'\xe5\x91\xa8\xe6\x9c\x9f'),
    # 偏移量? -> 偏移量
    (b'\xe5\x81\x8f\xe7\xa7\xbb\xe9\x87\x8f\xef\xbc\x3f', b'\xe5\x81\x8f\xe7\xa7\xbb\xe9\x87\x8f'),
]

fixed_count = 0
for old_bytes, new_bytes in patterns_to_fix:
    while old_bytes in data:
        idx = data.find(old_bytes)
        data[idx:idx+len(old_bytes)] = new_bytes
        fixed_count += 1
        print(f'Fixed pattern at {idx}')

if fixed_count == 0:
    print('No patterns found to fix - checking for 0x3f in problematic areas')
    # Search all the docstring/comment areas for 0x3f that could be corrupted
    # Let's find the docstring area
    doc_start = data.find(b'"""')
    doc_end = data.find(b'"""', doc_start + 3)
    print(f'Docstring: {doc_start} to {doc_end}')
    
    # Check for 0x3f after valid UTF-8 Chinese sequences
    count_q = data.count(b'\x3f', doc_start, doc_end)
    print(f'Found {count_q} question marks in docstring')
    
    # Show each one with context
    pos = doc_start
    while True:
        idx = data.find(b'\x3f', pos, doc_end)
        if idx < 0:
            break
        context = data[max(0,idx-10):idx+5]
        print(f'  ? at {idx}: hex={context.hex()}')
        pos = idx + 1

# Also check the main parameter section comments
param_section = data.find(b'# ============================================================')
print(f'\nParam section at {param_section}')
if param_section >= 0:
    count_q2 = data.count(b'\x3f', param_section, param_section + 500)
    print(f'Found {count_q2} question marks in param section')

# Fix line 103: missing double quote before closing )
# Find: ef bc 89 29 0d 0a (）\r\n) -> need: ef bc 89 22 29 0d 0a (）"\r\n)
# But the pattern might be different now
# Let's check by searching around the code section
idx_line103 = data.find(b'pr_info("')
while idx_line103 >= 0:
    end_of_line = data.find(b'\r\n', idx_line103)
    if end_of_line > 0:
        line = data[idx_line103:end_of_line]
        # Check if it's the line we need
        if b'\xe8\x87\xaa\xe5\x8a\xa8\xe8\xa3\x85\xe9\x85\x8d' in line:  # 自动装配
            print(f'Found auto assembly line at {idx_line103}')
            print(f'Line content: {line}')
            # Check if it ends with ）) or ）")
            if line.endswith(b'\xef\xbc\x89)'):
                print('Missing double quote - fixing...')
                # Insert " before )
                fix_idx = idx_line103 + len(line) - 1  # position of )
                data[fix_idx:fix_idx] = b'"'
                print('Fixed!')
                break
            elif line.endswith(b'\xef\xbc\x89")'):
                print('Already has double quote - OK')
            else:
                print(f'Unexpected ending: {line[-10:]}')
    idx_line103 = data.find(b'pr_info("', idx_line103 + 1)

with open(path, 'wb') as f:
    f.write(data)

print(f'\nTotal fixes: {fixed_count}')
print('Done')
