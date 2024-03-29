import sys
import discord
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none
import torch
import os
import MeCab
import sys
import subprocess
import alkana
import re
import urllib.request, urllib.error
import soundfile as sf
import glob
import emoji
from urllib.parse import urlparse
import unicodedata as ud
import asyncio
import time
from espnet_model_zoo.downloader import ModelDownloader

d = ModelDownloader()  # <module_dir> is used as cachedir by default

TOKEN = "TOKEN"

mypath = os.getcwd() + "/"
dict = []
state = []

prefix = "!"
__AUTHOR = "KuronekoServer"

alkana.add_external_data('./additional_dictionaly.csv')

limit_timeout = 8
timeout_time = 60
spam_threshould = 300

def wakati(sentence):
    wakati = MeCab.Tagger("-Owakati")
    words = wakati.parse(sentence).split()
    return words

def is_meaning(sen, words):
    spam = 0
    for word in words:
        if word[:1] == "死":
            spam += 70
        cnt = sen.count(word)
        spam += cnt
    spam += len(words)
    return spam

def exec_cmd(cmd):
    try:
        result = subprocess.run(cmd, shell=True, check=True,stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        return result.stdout
    except subprocess.CalledProcessError:
        return -1

def newfile(filename):
    exec_cmd("touch " + filename)
    f = open(filename, 'w')
    f.close()

class glo:
    sr = 48000
    _format = "WAV"
    subtype = 'PCM_16'
    pass

class dictionaly():
    def __init__(self, sid):
        self.moto = []
        self.sid = sid
        self.henkan = []
        if os.path.exists(mypath + "lib/" + str(self.sid) + ".lib") == False:
             newfile(mypath + "lib/" + str(self.sid) + ".lib")
        f = open(mypath + "lib/" + str(self.sid) + ".lib", 'r')
        for i in range(sum([1 for _ in open(mypath + "lib/" + str(self.sid) + ".lib")])):
            if f.readline() == "":
                break
            sen = f.readline()
            words = sen.split("==========>")
            if len(words) == 1:
                break
            self.moto.append(words[0])
            self.henkan.append(words[1].replace("\n", ""))
        f.close()

    def apliy(self, sen):
        for i in range(len(self.moto)):
            sen = sen.replace(self.moto[i], self.henkan[i])
        return sen

    def dict_from_file(self, fn):
        f = open(fn, 'r')
        for i in range(sum([1 for _ in open(fn)])):
            if f.readline() == "":
                break
            sen = f.readline()
            words = sen.split("==========>")
            if len(words) == 1:
                break
            self.add_word(words[0], words[1].replace("\n", ""))
        f.close()
        return

    def get_word(self, word):
        if len(self.moto) == 0:
            return -1
        for i in range(len(self.moto)):
            if self.moto[i] == word:
                return self.henkan[i]
        return 0

    def add_word(self, moto, henkan):
        if moto in "==========>" or henkan in "==========>":
            return -1
        if moto in "==========>" and henkan in "==========>":
            return -1
        for s in self.moto:
            if s == moto:
                return -1
        self.moto.append(moto)
        self.henkan.append(henkan)
        return

    def delete_word(self, word):
        if len(self.moto) == 0:
            return -1
        for i in range(len(self.moto)):
            if self.moto[i] == word:
                del(self.moto[i])
                del(self.henkan[i])
        return 0

    def apliy_file(self):
        if len(self.moto) == 0:
            return -1
        if os.path.exists(mypath + "lib/" + str(self.sid) + ".lib") == False:
             newfile(mypath + "lib/" + str(self.sid) + ".lib")
        f = open(mypath + "lib/" + str(self.sid) + ".lib", "w")
        f.write("dummy==========>dummy" + "\n")
        for i in range(len(self.moto)):
            f.write(self.moto[i] + "==========>" + self.henkan[i] + "\n")

class server():
    def __init__(self, sid):
        self.sid = sid
        self.vol = 100
        self.mod = 1
        self.ryaku = 50
        self.prefix = "!"
        self.cid = ""
        self.rate = 48000
        self.timeout = 0
        self.time = 0
        self.nowtime = 0
        self.spam = 0
        self.is_connect = 0
        if os.path.exists(mypath + "state/" + str(self.sid) + ".ini") == False:
            newfile(mypath + "state/" + str(self.sid) + ".ini")
            f = open(mypath + "state/" + str(self.sid) + ".ini", 'w')
            f.write("100,1,50,!")
            f.close()
        f = open(mypath + "state/" + str(self.sid) + ".ini", 'r')
        for i in range(sum([1 for _ in open(mypath + "state/" + str(self.sid) + ".ini")])):
            sen = f.readline()
            words = sen.split(",")
            self.vol = int(words[0])
            self.mod = int(words[1])
            self.ryaku = int(words[2])
        f.close()

    def apliy():
        f = open(mypath + "state/" + str(self.sid) + ".ini", 'w')
        f.write(str(self.vol) + "," + str(self.mod) + "," + str(self.ryaku) + "," + str(self.prefix))
        f.close()

def check_dict(sid):
    for i in range(len(dict)):
        if dict[i].sid == int(sid):
            return i
    return -1

def check_state(sid):
    for i in range(len(state)):
        if state[i].sid == sid:
            return i
    return -1

def get_connect_num():
    num = 0
    for i in range(len(state)):
        if state[i].is_connect == 1:
            num += 1
    return num

def ryaku(sentence, i):
    if len(sentence) >= state[i].ryaku:
         sentence = sentence[:state[i].ryaku]
         sentence += " 以下略"
    return sentence

text2speech_1 = Text2Speech.from_pretrained(
    model_tag=str_or_none('kan-bayashi/tsukuyomi_full_band_vits_prosody'),
    vocoder_tag=str_or_none('parallel_wavegan/jsut_multi_band_melgan.v2'),
    device="cpu",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

text2speech_1_slow = Text2Speech.from_pretrained(
    model_tag=str_or_none('kan-bayashi/tsukuyomi_full_band_vits_prosody'),
    vocoder_tag=str_or_none('parallel_wavegan/jsut_multi_band_melgan.v2'),
    device="cpu",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=2,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

text2speech_2 = Text2Speech.from_pretrained(
     **d.download_and_unpack('./tts_train_tacotron2_osaka_raw_phn_jaconv_pyopenjtalk_train.loss.ave.zip'),
    vocoder_tag=str_or_none('none'),
    device="cpu",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

text2speech_3 = Text2Speech.from_pretrained(
    model_tag=str_or_none("kan-bayashi/jsut_full_band_vits_prosody"),
    vocoder_tag=str_or_none('none'),
    device="cpu",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=1,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

text2speech_2_slow = Text2Speech.from_pretrained(
     **d.download_and_unpack('./tts_train_tacotron2_osaka_raw_phn_jaconv_pyopenjtalk_train.loss.ave.zip'),
    vocoder_tag=str_or_none('none'),
    device="cpu",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=2,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

text2speech_3_slow = Text2Speech.from_pretrained(
    model_tag=str_or_none("kan-bayashi/jsut_full_band_vits_prosody"),
    vocoder_tag=str_or_none('none'),
    device="cpu",
    # Only for Tacotron 2 & Transformer
    threshold=0.5,
    # Only for Tacotron 2
    minlenratio=0.0,
    maxlenratio=10.0,
    use_att_constraint=False,
    backward_window=1,
    forward_window=3,
    # Only for FastSpeech & FastSpeech2 & VITS
    speed_control_alpha=2,
    # Only for VITS
    noise_scale=0.333,
    noise_scale_dur=0.333,
)

def tts2wav(sentence, mod, slow, rate):
    with torch.no_grad():
      if mod == 1:
            rate = 48000
            if slow == 0:
                wav = text2speech_1(sentence)["wav"]
            else:
                wav = text2speech_1_slow(sentence)["wav"]
      if mod == 3:
            rate = 48000
            if slow == 0:
                wav = text2speech_3(sentence)["wav"]
            else:
                wav = text2speech_3_slow(sentence)["wav"]
      if mod == 2:
            rate = 24000
            if slow == 0:
                wav = text2speech_2(sentence)["wav"]
            else:
                wav = text2speech_2_slow(sentence)["wav"]
      sf.write(mypath + "tts/" + sentence + ".wav", wav, rate, format=glo._format, subtype=glo.subtype)
      return mypath + "tts/" + sentence + ".wav"
      if os.path.exists(mypath + "tts/" + sentence + ".wav") == False:
            return -1

def check_url(url):
    flag = True
    p = urlparse(url)
    query = urllib.parse.quote_plus(p.query, safe='=&')
    url = '{}://{}{}{}{}{}{}{}{}'.format(
        p.scheme, p.netloc, p.path,
        ';' if p.params else '', p.params,
        '?' if p.query else '', query,
        '#' if p.fragment else '', p.fragment)
    try:
        f = urllib.request.urlopen(url)
        f.close()
    except urllib.request.HTTPError:
        flag = False
    return flag

def edit_sentence(sen, i, j):
    if sen[:1] == "|" and sen[len(sen)-1:] in "|":
        print("slow")
        sen.replace("|", "")
        slow = 1
    else:
        slow = 0
    if re.search("http://", sen) != None or re.search("https://", sen) != None:
        if check_url(sen.split(" ")[0]) != False:
             sen = "URLが送信されました"
        else:
             sen = "URLが送信されました"
    else:
        if re.search("<:", sen) != None and re.search(">", sen) != None:
            sen = "スタンプが添付されました"
    sen = emoji.demojize(sen)
    sen = ryaku(sen, i)
    alp = wakati(sen)
    for i in range(len(alp)):
        if alkana.get_kana(alp[i]) != None:
            sen = sen.replace(alp[i], alkana.get_kana(alp[i]))
    sen = dict[j].apliy(sen)
    print(alp)
    print(sen)
    return sen, slow

def remove_glob(pathname, recursive=True):
    for p in glob.glob(pathname, recursive=recursive):
        if os.path.isfile(p):
            os.remove(p)

client = discord.Client(allowed_mentions=discord.AllowedMentions.none())

@client.event

async def on_ready():
    print('------')
    print("TTSBot poweredby Eurobeat-Lover at YH")
    print("Bot Name:" + client.user.name)  # Botの名前
    print("Bot ID:" + str(client.user.id))  # ID
    print("library version:" + discord.__version__)  # discord.pyのバージョン
    print('------')
    await client.change_presence(activity=discord.Game(name=f"BOT利用数：{len(client.guilds)}サーバー"))

@client.event

async def on_message(message):
    if not os.path.exists("is_mkdir"):
        os.mkdir(mypath + "lib/")
        os.mkdir(mypath + "state/")
        os.mkdir(mypath + "tts/")
        newfile(mypath + "is_mkdir")
    if message.author.bot:
        return
    if check_dict(message.guild.id) != -1:
        array_dict_jump = check_dict(message.guild.id)
    else:
        num = len(dict)
        dict.append("")
        dict[num] = dictionaly(message.guild.id)
        array_dict_jump = num
    if check_state(message.guild.id) != -1:
        array_state_jump = check_state(message.guild.id)
    else:
        num = len(state)
        state.append("")
        state[num] = server(message.guild.id)
        array_state_jump = num
    if message.content == prefix + "jn":
        if message.author.voice is None:
            await message.channel.send("あなたはボイスチャンネルに接続していません。")
            return
        # ボイスチャンネルに接続する
        state[array_state_jump].cid = message.channel.id
        state[array_state_jump].is_connect = 1
        await message.author.voice.channel.connect()
        await message.channel.send("```接続しました。```")
        message.content = "接続しました"
    if re.search(prefix + "add", message.content) != None:
        sent = message.content
        sent = sent.replace(prefix + "add ","")
        sents = sent.split(" ")
        dict[array_dict_jump].add_word(sents[0], sents[1])
        dict[array_dict_jump].apliy_file()
        await message.channel.send("```辞書登録 " + sents[0] + " => " + sents[1] + "```")
        return
    if re.search(prefix + "remove", message.content) != None:
        sent = message.content
        sent = sent.replace(prefix + "remove ","")
        if not dict[array_dict_jump].delete_word(sent) == -1:
            await message.channel.send("単語削除 " + sent)
        dict[array_dict_jump].apliy_file()
        return
    if message.content == prefix + "help":
        await message.channel.send("```" + prefix + "help: コマンドの詳細を表示します\n" + prefix + "jn: ボイスチャンネルに接続します\n" + prefix + "add: 辞書に単語を追加します\n" + prefix + "remove: 辞書から単語を削除します\n" + prefix + "volume: 音量を変更します\n" + prefix + "ryaku: 読む長さを変更します\n" + prefix + "lv: ボイスチャンネルから切断します\n" + prefix + "about: ボットの詳細を表示します```")
    if re.search(prefix + "voice", message.content) != None:
        sent = message.content
        sent = sent.replace(prefix + "voice ","")
        if sent == "":
            await message.channel.send("```声 1:つくよみちゃん 2:? 3:JSUT```")
        num = int(sent)
        if num == 1:
            state[array_state_jump].rate = 48000
            state[array_state_jump].mod = 1
            await message.channel.send("```音声を変えました。```")
        elif num == 2:
            state[array_state_jump].rate = 24000
            state[array_state_jump].mod = 2
            await message.channel.send("```音声を変えました。```")
        elif num == 3:
            state[array_state_jump].rate = 24000
            state[array_state_jump].mod = 3
            await message.channel.send("```音声を変えました。```")
        return
    if re.search(prefix + "volume", message.content) != None:
        sent = message.content
        if sent == prefix + "volume":
            await message.channel.send("```音量 => " + str(state[array_state_jump].vol) + "```")
        sent = sent.replace(prefix + "volume ","")
        state[array_state_jump].vol = int(sent)
        await message.channel.send("```音量を変えました。 => " + str(state[array_state_jump].vol) + "```")
        state[array_state_jump].apliy()
        return
    if message.content == prefix + "download":
        await message.channel.send("```辞書ファイルをダウンロードします```")
        await message.channel.send(file=discord.File(mypath + "lib/" + str(message.guild.id) + ".lib"))
        return
    if message.content == prefix + "dict":
        if len(message.attachments) != 0:
            file = message.attachments[0]
            url = file.url
            type = file.filename.split(".")[1].lower()
            if type == "lib":
                exec_cmd("curl " + url + " > " + mypath + "userdict_" + str(message.guild.id) + ".lib")
                dict[array_dict_jump].dict_from_file(mypath + "userdict_" + str(message.guild.id) + ".lib")
                remove_glob(mypath + "userdict_" + str(message.guild.id) + ".lib")
                await message.channel.send("```辞書ファイルを適応しました```")
        else:
            await message.channel.send("```辞書ファイルが添付されてません```")
        return
    if re.search(prefix + "ryaku", message.content) != None:
        sent = message.content
        if sent == prefix + "ryaku":
            await message.channel.send("```略の長さ => " + str(state[array_state_jump].ryaku) + "```")
            return
        sent = sent.replace(prefix + "ryaku ","")
        state[array_state_jump].ryaku = int(sent)
        await message.channel.send("```略の長さを変えました。 => " + str(state[array_state_jump].ryaku) + "```")
        state[array_state_jump].apliy()
        return
    if message.content == prefix + "lv":
        if message.guild.voice_client is None:
            await message.channel.send("```接続していません。```")
            return
        state[array_state_jump].cid = ""
        state[array_state_jump].spam = 0
        state[array_state_jump].is_connect = 0 
        await message.guild.voice_client.disconnect()
        await message.channel.send("```切断しました。```")
    if message.content == prefix + "ping":
        await message.channel.send("```botのレイテンシー: " + str(round(client.latency*1000)) + "ms```")
        return
    if message.content == prefix + "about":
        await message.channel.send("```こんにちは！読み上げちゃんです！\nこのBOTは" + __AUTHOR + "によって一から作られました\n協力してくれた人\n------------------\nYH,黒猫ちゃん\n------------------\nサポートサーバー\nhttps://discord.gg/Y6w5Jv3EAR```")
        return
    if state[array_state_jump].cid == message.channel.id:
        if state[array_state_jump].timeout != 1:
            if state[array_state_jump].spam == 1:
                await message.channel.send("```スパムかな？```")
            elif state[array_state_jump].spam == 2:
                await message.channel.send("```やめようねｗ？```")
            nowtime = time.time()
            state[array_state_jump].time =  nowtime - state[array_state_jump].nowtime
            print(is_meaning(message.content, wakati(message.content)))
            print(state[array_state_jump].spam)
            if state[array_state_jump].spam > 2:
                state[array_state_jump].timeout = 1
                state[array_state_jump].spam = 0
                await message.channel.send("```タイムアウトが有効になりました\nすこし落ち着きましょうｗby黒猫```")
            if is_meaning(message.content, wakati(message.content)) > spam_threshould:
                state[array_state_jump].spam += 1
                return
            else:
                state[array_state_jump].spam = 0
            if state[array_state_jump].time < limit_timeout and is_meaning(message.content, wakati(message.content)) > spam_threshould:
                state[array_state_jump].spam += 1
                return
            else:
                state[array_state_jump].spam = 0
            state[array_state_jump].nowtime = nowtime
            sen = message.content
            if len(message.attachments) != 0:
                file = message.attachments[0]
                type = file.filename.split(".")[1].lower()
                if type == "webp" or type == "jpg" or type == "png" or type == "gif" or type == "bmp":
                    sen = "画像が添付されました"
                elif type == "mp3" or type == "m4a" or type == "wav" or type == "ogg" or type == "flac" or type == "aac":
                    sen = "音声ファイルが添付されました"
                elif type == "mov" or type == "mp4" or type == "mkv" or type == "avi":
                    sen = "動画ファイルが添付されました"
                elif type == "txt":
                    sen = "テキストファイルが添付されました"
                else:
                    sen = "ファイルが添付されました"
            sents = sen.split("\n")
            for s in sents:
                if message.mentions is not None:
                    for mention in re.findall(r"@(everyone|here|[!&]?[0-9]{17,20})", message.content):
                        if mention.replace("@", "") == "everyone":
                            s = s.replace("@everyone", "@ everyone")
                        elif mention.replace("@", "") == "here":
                            s = s.replace("@here", "@ here")
                        else:
                            userId = re.match(r"[0-9]+", str(mention.replace("!", ""))).group()
                            try:
                                u = await client.fetch_user(int(userId))
                            except:
                                sen = sen.replace(mention, "@" + str(userId))
                            else:
                                if userId.isdigit():
                                    s = s.replace(mention, u.name)
                    else:
                        s = discord.utils.escape_mentions(s)
                if re.search("<#", sen) != None and re.search(">", sen) != None:
                    cn = client.get_channel(int(s[2:].replace(">", "")))
                    s = cn.name
                sen, slow = edit_sentence(s, array_state_jump, array_dict_jump)
                message.guild.voice_client.play(discord.PCMVolumeTransformer(discord.FFmpegPCMAudio(tts2wav(sen,state[array_state_jump].mod,slow,state[array_state_jump].rate)), volume=(state[array_state_jump].vol / 100)))
    #            await message.channel.send(sen)
                await asyncio.sleep(len(sen)/10)
                remove_glob(mypath + "tts/" + sen + ".wav")
        else:
            if state[array_state_jump].nowtime + (60*timeout_time) < time.time():
                state[array_state_jump].timeout = 0
                await message.channel.send("```タイムアウトが解除されました```")

    await client.change_presence(activity=discord.Game(name="導入数：" + str(len(client.guilds)) + " | 接続数：" + str(get_connect_num())))

client.run(TOKEN)
