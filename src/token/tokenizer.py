
import tiktoken
from transformers import BertTokenizer

# 太重了 100256
# Encoder = tiktoken.get_encoding('cl100k_base')

# 太轻了 但好在可以自适应适配，21128
# Encoder = BertTokenizer.from_pretrained("bert-base-chinese")

# 自定义 21128 + n

# 文本素材
vocab_a = [
    # 凡人修仙
    "韩立", "墨彩环", "厉飞雨", "南宫婉", "紫灵", "银月", "古或今", "轮回殿主", "元瑶", "董萱儿",
    "辛如音", "周姓青年", "风希人", "青元子", "古烈", "冰凤", "啼魂兽", "花无缺", "虚天殿主", "宝花",
    "青元剑诀", "大衍决", "吞灵决", "紫罗极焰", "玄天斩灵剑", "青竹蜂云剑", "小极宫剑阵", "两仪微尘阵", "太乙青灵符", "化形丹",
    "破禁珠", "敛息草", "虚天鼎", "如意瓶", "五行珠", "七幻迷神铃", "千幻阴灵大阵", "兜率紫焰", "雷元刀", "玄阴聚兽幡",
    "炼气期", "筑基期", "结丹期", "元婴期", "化神期", "炼虚期", "合体期", "大乘期", "真仙", "太乙境",
    "大罗境", "人族", "妖族", "鬼族", "灵族", "古魔族", "灰界生物", "天渊族", "鳞族", "海兽",
    "黄枫谷", "落云宗", "小极宫", "灵雾岛", "乱星海", "慕兰草原", "昆吾山", "西漠", "冰原", "灵界",
    "灰界", "大罗天域", "积鳞空境", "仙宫", "万宝楼", "虚天殿", "轮回殿", "元刹海", "冥寒仙府", "古或今洞府",
    "修仙", "灵根", "灵液", "丹药", "符箓", "禁制", "法宝", "灵矿", "斗法", "秘境",
    "机缘", "傀儡", "灵宠", "灵草", "灵虫", "传送阵", "灵界通道", "血祭", "夺舍", "封印",

    # 斗破苍穹
    "萧炎", "萧薰儿", "美杜莎", "药尘", "纳兰嫣然", "云韵", "小医仙", "海波东", "古元", "魂天帝",
    "萧战", "萧鼎", "萧厉", "雅妃", "青鳞", "紫妍", "天火尊者", "风尊者", "雷尊者", "玄空子",
    "斗之气", "斗者", "斗师", "大斗师", "斗灵", "斗王", "斗皇", "斗宗", "斗尊", "斗圣",
    "斗帝", "灵魂力量", "斗技", "功法", "异火", "丹塔", "加玛帝国", "魔兽山脉", "黑岩城", "塔戈尔大沙漠",
    "迦南学院", "天焚炼气塔", "黑角域", "古族", "魂族", "石族", "炎族", "雷族", "药族", "灵族",
    "萧族", "骨灵冷火", "青莲地心火", "陨落心炎", "三千焱炎火", "九幽金祖火", "净莲妖火", "虚无吞炎", "海心焰", "火云水炎",
    "融灵丹", "筑基灵液", "生源丹", "六品融血丹", "七幻青灵涎", "阴阳玄龙丹", "天魂融血丹", "八品复魂丹", "九品玄丹", "黄泉妖圣丹",
    "斗气大陆", "远古八族", "封印", "空间戒指", "纳戒", "紫晶翼狮王", "天妖凰族", "九幽地冥蟒", "太古虚龙", "蛇人部落",
    "萧战的家族", "萧战的府邸", "迦南学院内院", "黑角域势力", "中州", "远古遗迹", "虚无空间", "灵魂殿", "血宗", "丹会",

    # 剑来
    "陈平安", "宁姚", "裴钱", "齐静春", "崔瀺", "文圣", "亚圣", "礼圣", "道老二", "左右",
    "阿良", "老剑条", "阮秀", "刘羡阳", "李宝瓶", "马苦玄", "隋右边", "杨老头", "宋集薪", "姚近之",
    "剑气长城", "骊珠洞天", "落魄山", "桐叶洲", "中土神洲", "宝瓶洲", "北俱芦洲", "流霞洲", "南婆娑洲", "西玄洲",
    "文胆", "武运", "剑心", "符箓", "道法", "拳法", "剑术", "阵法", "炼神术", "养剑术",
    "飞升境", "玉璞境", "仙人境", "元婴境", "金丹境", "指玄境", "天象境", "武夫境", "纯武夫", "大剑仙",
    "儒家", "道家", "佛家", "兵家", "法家", "纵横家", "阴阳家", "农家", "杂家", "墨家",
    "山河印", "养剑葫", "白玉京", "五彩楼船", "落魄钟", "雷池剑匣", "青冥镜", "黄庭国玺", "太平玉玺", "镇妖印",
    "土地神", "山神", "河神", "水神", "树神", "夜游神", "夜游神", "火神", "谷神", "井神",
    "问心局", "大骊礼部", "桐叶宗", "正阳山", "落魄山", "剑气长城隐官一脉", "披云山", "龙霄宫", "雪宫", "白草岭",
    "一洲之地", "浩然天下", "蛮荒天下", "白玉京", "剑气长城之战", "十境武夫", "三教圣人", "十四境大修士", "练气士", "神道"
]

vocab_b = [
    # 动词
    "斩击", "劈砍", "刺挑", "踢踹", "掌掴", "拳打", "肘击", "膝顶", "横扫", "竖斩",
    "剑指", "刀劈", "枪挑", "鞭抽", "箭射", "雷轰", "火焚", "冰冻", "风卷", "土埋",
    "奔跑", "疾奔", "跳跃", "腾跃", "飞翔", "翱翔", "飘移", "瞬移", "穿梭", "滑行",
    "施法", "念咒", "结印", "画符", "布阵", "召唤", "释放", "凝聚", "消散", "炼化",
    "交谈", "辩论", "争论", "询问", "回答", "劝说", "告诫", "呵斥", "怒骂", "低语",
    "探寻", "搜寻", "挖掘", "勘探", "查看", "窥视", "洞察", "寻觅", "追踪", "侦查",
    "修炼", "修行", "冥想", "顿悟", "突破",
    # 神情
    "欣喜", "欢喜", "愉悦", "畅快", "欢畅", "开怀", "狂喜", "大喜", "喜出望外", "眉开眼笑",
    "喜笑颜开", "笑容满面", "笑逐颜开", "欣喜若狂", "心花怒放",
    "愤怒", "恼怒", "气愤", "盛怒", "暴怒", "怒火中烧", "怒发冲冠", "横眉怒目", "怒目而视", "咬牙切齿",
    "暴跳如雷", "大发雷霆", "恼羞成怒", "气急败坏", "义愤填膺",
    "悲伤", "悲痛", "哀伤", "哀愁", "悲戚", "黯然神伤", "泣不成声", "肝肠寸断", "悲痛欲绝", "泪如雨下",
    "潸然泪下", "愁眉苦脸", "愁眉不展", "郁郁寡欢", "黯然销魂",
    "惊恐", "惊慌", "惊吓", "惶恐", "惊悚", "胆战心惊", "心惊肉跳", "毛骨悚然", "面如土色", "大惊失色",
    "惊慌失措", "手足无措", "惶恐不安", "魂飞魄散", "六神无主",
    "疑惑", "狐疑", "猜疑", "猜忌", "困惑", "大惑不解", "满腹狐疑", "半信半疑", "迷惑不解", "一头雾水",
    "平静", "淡定", "从容", "镇定", "泰然", "面不改色", "神色自若", "从容不迫", "泰然自若", "心平气和",
    "轻蔑", "轻视", "鄙视", "鄙夷", "不屑", "嗤之以鼻", "不屑一顾", "目空一切", "傲睨万物", "颐指气使",
    "诧异", "惊愕", "惊讶", "讶异", "怔愣", "瞠目结舌", "目瞪口呆", "呆若木鸡", "若有所思", "神情专注"
]


Encoder = BertTokenizer.from_pretrained("bert-base-chinese")
Encoder.add_tokens(vocab_a)
Encoder.add_tokens(vocab_b)

Encoder.add_tokens(["\n", "\\n", "“", "”", "‘", "’", "......", "..."])
Encoder.add_special_tokens({'eos_token': '[EOS]'})

# 自定义 tokenizer
Encoder.save_pretrained("src/token/Encoder")

print("Encoder Loading -> Succeed", len(Encoder))
