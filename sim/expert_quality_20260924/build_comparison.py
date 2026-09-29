"""Build a local, evidence-linked review page from completed numerical audits.

This presentation does not change acceptance, choose thresholds, or score video.
"""
import argparse
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def collect(key, label, description, report_dir, review_dir, free_report_name='free_motion_report.json'):
    quality_path = ROOT / 'results' / report_dir / 'quality_report.json'
    review_path = ROOT / 'results' / review_dir / 'review_manifest.json'
    quality = json.loads(quality_path.read_text())
    free_path=quality_path.with_name(free_report_name)
    free=json.loads(free_path.read_text())
    if any(free['provenance']['artifact_sha256'].get(name)!=quality['artifact_sha256'][name]
           for name in ('expert_trace.jsonl','outcome.json','control_commands.json')):
        raise ValueError('free-motion audit uses different evidence: '+key)
    review = json.loads(review_path.read_text())
    media=json.loads((review_path.parent/'media_verification.json').read_text())
    if media.get('source_bindings_passed') is not True or media.get('manifest_sha256')!=digest(review_path):
        raise ValueError('media verification absent or stale: '+key)
    decoded={item['clip']:item for item in media['videos']}
    for name in ('outcome.json', 'expert_trace.jsonl', 'expert_replay.npz'):
        expected = quality['artifact_sha256'][name]
        bindings = [v for k, v in review['source_sha256'].items() if k.endswith('/'+name)]
        if bindings != [expected]:
            raise ValueError('video and quality input differ: '+key+'/'+name)
    if review['playback_speed'] != 1. or review['no_interpolation'] is not True:
        raise ValueError('comparison requires recorded 1x playback')
    clips = {}
    for clip in review['clips']:
        video = review_path.parent / clip['video']
        checked=decoded.get(clip['clip_id'],{})
        if (checked.get('decoded_full') is not True or checked.get('sha256')!=clip['video_sha256']
                or checked.get('frame_count')!=clip['frame_count']):
            raise ValueError('clip has not passed full decoding: '+clip['clip_id'])
        if digest(video) != clip['video_sha256']:
            raise ValueError('video digest mismatch: '+str(video))
        clips[clip['clip_id']] = {'video': str(video.relative_to(ROOT)),
                                 'duration_s': clip['duration_s']}
    required_clips={point+'_'+view for point in ('R1','L1','R5') for view in ('posture','robotcam')}
    if not required_clips.issubset(clips):
        raise ValueError('comparison missing target/view clips: '+str(sorted(required_clips-set(clips))))
    measured = quality['motion']['global']['measured']
    contact = quality['contact']['legacy']['points']
    expected_order=[side+str(number) for number in range(1,6) for side in ('R','L')]
    if [p['target'] for p in contact]!=expected_order:
        raise ValueError('full ten-point comparison requires the same complete point order: '+key)
    return dict(key=key, label=label, description=description,
                quality=str(quality_path.relative_to(ROOT)), quality_sha256=digest(quality_path),
                free_report=str(free_path.relative_to(ROOT)), free_report_sha256=digest(free_path),
                free_command_jerk=free['actual_windows']['jerk_rad_s3']['peak']['value'],
                free_command_status=free['status'],
                review=str(review_path.relative_to(ROOT)), review_sha256=digest(review_path),
                clips=clips, hard_failures=quality['hard_failures'],
                simulation_decision=quality['simulation_quality_decision'],
                travel=measured['total_travel_rad'], per_joint=measured['travel_rad'],
                reversals=measured['reversal_count'],
                duration=quality['motion']['sampling']['duration_s'],
                elbow_above_shoulder_mm=1000*measured['posture']['max_elbow_above_shoulder_m'],
                contact_force_peak=max(p['max_physical_force_N'] for p in contact),
                min_hold=min(p['continuous_front_contact_hold_s'] for p in contact),
                completed_points=sum(p['accepted'] for p in contact),
                jerk_raw=max(measured['derivatives']['raw']['jerk']['max_abs_per_joint']),
                acceleration_raw=max(measured['derivatives']['raw']['acceleration']['max_abs_per_joint']),
                jerk_filtered=max(measured['derivatives']['filtered']['jerk']['max_abs_per_joint']),
                transitions=[dict(route=t['from_target']+' → '+t['to_target'],
                    travel=t['measured']['total_travel_rad'],
                    start_time_s=t['start_time_s'],end_time_s=t['end_time_s'])
                    for t in quality['motion']['transitions']])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    data=[collect(**v) for v in config['variants']]
    for item in data:
        item['travel_reduction_pct']=100*(1-item['travel']/data[0]['travel'])
    table_rows=[('累计转角 · rad','travel',2),('方向反转 · 次（>1°）','reversals',0),
                ('累计转角相对原版减少 · %','travel_reduction_pct',2),
                ('全程模拟时间 · s','duration',2),('最高肘肩差 · mm','elbow_above_shoulder_mm',1),
                ('最大总物理接触力 · N','contact_force_peak',3),('最短连续有效 HOLD · s','min_hold',2),
                ('通过旧物理检查的点数 / 10','completed_points',0),
                ('实测加速度原始峰值 · rad/s²','acceleration_raw',3),
                ('实测 jerk 原始峰值 · rad/s³','jerk_raw',1),
                ('实测 jerk 0.1s 滤波峰值 · rad/s³','jerk_filtered',1),
                ('自由移动命令 jerk 峰值 · rad/s³','free_command_jerk',3)]
    table=''.join('<tr><th>'+html.escape(name)+'</th>'+''.join(f'<td>{d[key]:.{places}f}</td>' for d in data)+'</tr>' for name,key,places in table_rows)
    cards=''.join(f'<article><h3>{html.escape(d["label"])}</h3><p>{html.escape(d["description"])}</p><video id="v{i}" controls preload="metadata" playsinline></video><p id="t{i}" class="muted"></p><a href="{html.escape(d["quality"])}">数值报告</a> · <a href="{html.escape(d["review"])}">帧与物理记录映射</a></article>' for i,d in enumerate(data))
    heads=''.join('<th>'+html.escape(d['label'])+'</th>' for d in data)
    status='<tr><th>旧物理/姿态硬门槛</th>'+''.join('<td>'+('通过' if not d['hard_failures'] else html.escape('拒绝：'+'; '.join(d['hard_failures'])))+'</td>' for d in data)+'</tr>'
    decision_labels={'review_draft_preference_exceedances':'新草案有超限，待复核及选择',
        'reject_hard_failure':'硬失败，拒绝','pending_evaluation':'证据不足，待评估',
        'reject_selected_preference':'未满足已选偏好，拒绝',
        'passes_selected_engineering_profile':'已选工程指标通过',
        'passes_legacy_gates_draft_preferences_within_bounds':'旧硬门槛及草案范围内；仍待选标准'}
    decisions='<tr><th>新的动作质量偏好</th>'+''.join('<td>'+html.escape(decision_labels.get(d['simulation_decision'],d['simulation_decision']))+'</td>' for d in data)+'</tr>'
    notes=''.join('<li>'+html.escape(note)+'</li>' for note in config['notes'])
    routes=[t['route'] for t in data[0]['transitions']]
    if len(routes)!=9 or any([t['route'] for t in d['transitions']]!=routes for d in data):
        raise ValueError('comparison requires identical nine transition intervals')
    transitions=''.join('<tr><th>'+html.escape(route)+'</th>'+''.join(
        f'<td>{d["transitions"][i]["travel"]:.3f} rad <small>({d["transitions"][i]["start_time_s"]:.2f}–{d["transitions"][i]["end_time_s"]:.2f}s)</small></td>'
        for d in data)+'</tr>' for i,route in enumerate(routes))
    payload=json.dumps(data,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>听诊专家动作对比</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f5f7fa;color:#182a3b;margin:0}main{max-width:1500px;margin:auto;padding:32px}h1{font-size:30px;margin:0 0 10px}h2{font-size:22px;margin-top:32px}h3{font-size:18px;margin:0}p{margin:10px 0}.lead{max-width:1050px}.muted{color:#536778;font-size:14px}.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}article,.panel{background:white;border:1px solid #d8e0e8;border-radius:12px;padding:18px}article p{min-height:50px}video{width:100%;display:block;background:#e4e8ec;border-radius:6px}button,select{font:inherit;padding:7px 12px;margin:5px;border:1px solid #8ba2b6;border-radius:6px;background:white;color:#162c40}button{background:#155c96;color:white;cursor:pointer}table{border-collapse:collapse;width:100%;font-size:15px}th,td{border-bottom:1px solid #dde4eb;padding:10px;text-align:left;overflow-wrap:anywhere}th{font-weight:600}a{color:#155c96}.bars{display:grid;grid-template-columns:repeat(7,1fr);gap:14px}.bar{height:12px;margin:7px 0;background:#527890;border-radius:2px}.bar.a{background:#19847a}.bar.b{background:#b47a27}.legend{display:flex;gap:20px;flex-wrap:wrap}@media(max-width:950px){.cards{grid-template-columns:1fr}.bars{grid-template-columns:repeat(4,1fr)}main{padding:18px}}</style>
<main><p class="muted">WAM · Isaac Sim · 2026-09-24 · 仿真动作比较</p><h1>减少换点时的关节扭动</h1>
<p class="lead">同一组十个点、相同点序、同一力控制器。视频按原速播放，来自真实物理运行保存的刚体姿态；相机保留已确认的向下 7°、上移 5 cm。对比片段选取 R1、L1、R5，全十点数值评估见下表。</p>
<div class="panel"><label>听诊点 <select id="point"><option>L1</option><option>R1</option><option>R5</option></select></label><label>视角 <select id="view"><option value="posture">全臂观察</option><option value="robotcam">已确认的相机</option></select></label><button id="play">从头同时原速播放</button><button id="pause">全部暂停</button><p class="muted">各方案从该点动作开始同时播放。较短片段结束后停在末帧；不拉伸时间、不补假动作。精确接触时长和高频抖动以 240 Hz 数值记录为准。</p></div>
<div class="cards">CARDS</div><h2>全十点实测</h2><div class="panel" style="overflow:auto"><table><thead><tr><th>指标</th>HEADS</tr></thead><tbody>TABLE STATUS DECISIONS</tbody></table><p class="muted">累计转角是七个关节绝对转动量之和，并非某个关节旋转了这么多圈。raw jerk 对采样和求解器噪声敏感；滤波值仅作补充，不能掩盖跳变。新偏好阈值尚未由用户选定，旧硬门槛通过并不自动成为最终专家标准。</p></div>
<h2>每个关节实际转动量</h2><div class="panel"><p class="legend">LEGEND</p><div class="bars" id="bars"></div><p class="muted">各组柱条使用相同 rad 比例尺。路线不变但执行更快的方案，不应被解释为减少关节转动。</p></div>
<h2>每次换点的实际转角</h2><div class="panel" style="overflow:auto"><table><thead><tr><th>换点</th>HEADS</tr></thead><tbody>TRANSITIONS</tbody></table><p class="muted">统一窗口为离开上一点最后一个 HOLD 样本后，直到下一点第一个 HOLD 样本；包含释放、换点与重新接触。括号为各自物理记录的时间范围。</p></div>
<h2>本轮结论与边界</h2><div class="panel"><ul>NOTES</ul><p><a href="MODEL_OPTIONS_ZH.md">其他专家生成器与独立模型评审</a> · <a href="RESULTS_ZH.md">完整结果与复现说明</a> · <a href="quality_profile.json">待选工程偏好</a></p></div>
<script>const data=DATA;function load(){data.forEach((d,i)=>{const key=document.querySelector('#point').value+'_'+document.querySelector('#view').value,c=d.clips[key],v=document.querySelector('#v'+i);v.pause();v.src=c.video;v.playbackRate=1;document.querySelector('#t'+i).textContent='该点片段 '+c.duration_s.toFixed(2)+' 秒 · 原速 1×';});}document.querySelector('#point').onchange=load;document.querySelector('#view').onchange=load;document.querySelector('#play').onclick=()=>{document.querySelectorAll('video').forEach(v=>{v.currentTime=0;v.playbackRate=1;v.play().catch(()=>{});});};document.querySelector('#pause').onclick=()=>document.querySelectorAll('video').forEach(v=>v.pause());const max=Math.max(...data.flatMap(d=>d.per_joint));for(let j=0;j<7;j++){const box=document.createElement('div');box.innerHTML='<strong>J'+(j+1)+'</strong>';data.forEach((d,i)=>{const row=document.createElement('div');row.title=d.label+' '+d.per_joint[j].toFixed(3)+' rad';const bar=document.createElement('div');bar.className='bar '+['','a','b'][i];bar.style.width=(100*d.per_joint[j]/max)+'%';row.append(bar);row.append(document.createTextNode(d.per_joint[j].toFixed(2)+' rad'));box.append(row);});document.querySelector('#bars').append(box);}load();</script></main></html>'''
    for token,value in {'CARDS':cards,'HEADS':heads,'TABLE':table,'STATUS':status,'DECISIONS':decisions,'NOTES':notes,'TRANSITIONS':transitions,
                         'LEGEND':''.join('<span>'+html.escape(d['label'])+'</span>' for d in data),'DATA':payload}.items():
        page=page.replace(token,value)
    (ROOT/'comparison.html').write_text(page)
    (ROOT/'results/comparison_summary.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('Wrote comparison.html and evidence-bound comparison_summary.json')


if __name__=='__main__':main()
