import re
import json
import ftfy


def get_lenegth(input_path):

    with open(input_path, 'r', encoding='utf-8') as f:
        total = 0
        row = 0
        for i, line in enumerate(f):
            try:
                text = json.loads(line.strip())['text']
                # append -> raw
                total += len(text)
                row += 1
            except Exception as e:
                print("Error encoding line:", e)
                continue
        print(row, total, total/row)


def format_text(input_path, output_path="src/data/demo-小说.txt"):

    chapter_pattern = re.compile(
        r'^(?:第[0-9一二三四五六七八九十百千]+卷\s+[^\s]+\s+)?'
        r'第([0-9一二三四五六七八九十百千]+)章\s*(.+)'
    )  # 新增章节识别
    clean_pattern = re.compile(
        r'\(看小说到网\).*|'
        r'\(未完待续.*?\)|'
        r'PS：.*|'
        r'------------|'
        r'^[\s\u3000]+|'
        r'[\s\u3000]+$|'
        r'（.*?）|'
        r'\(.*?\)'
    )

    current_title = "默认标题"
    content = []

    with open(input_path, 'r', encoding='utf-8') as f_in, \
            open(output_path, 'w', encoding='utf-8') as f_txt:

        for line in f_in:
            # 清理和修复文本
            line = ftfy.fix_text(line.strip())
            line = clean_pattern.sub('', line)

            if not line:
                continue

            # 章节识别
            if chapter_match := chapter_pattern.match(line):
                if content:
                    # 写入文本文件
                    f_txt.write('\n'.join(content))
                    content = []
                current_title = f"第{chapter_match.group(1)}章 {chapter_match.group(2)}"
            else:
                content.append(line)

        # 写入最后章节
        if content:
            f_txt.write('\n'.join(content))


format_text("src/source/剑来.txt")
