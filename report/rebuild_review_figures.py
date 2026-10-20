"""Render report-only category comparisons from saved metrics; no evaluation or relabeling."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
metrics = json.loads((ROOT.parent / "results/task4_safety/safety_metrics.json").read_text(encoding="utf-8"))
policies = ["sft", "dpo", "ppo", "grpo"]
labels = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
colors = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#999999"]
categories = sorted(metrics["policies"]["sft"]["by_category"], key=lambda c: (c.startswith("contrast_"), c))
names = [c.removeprefix("contrast_").replace("_", " ") for c in categories]
fig, axes = plt.subplots(1, 4, figsize=(6.5, 4.4), sharey=True)
for ax, policy in zip(axes, policies):
    rows = metrics["policies"][policy]["by_category"]
    left = [0.] * len(categories)
    for label, color in zip(labels, colors):
        values = [rows[c]["label_fractions"][label] for c in categories]
        ax.barh(range(len(categories)), values, left=left, color=color, height=.78)
        left = [a+b for a,b in zip(left,values)]
    ax.set_title(policy.upper(), fontsize=9)
    ax.set_xlim(0,1)
    ax.set_xticks([0,.5,1], labels=["0", ".5", "1"])
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.axhline(9.5, color="black", linewidth=.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
axes[0].set_yticks(range(len(categories)), labels=names, fontsize=8)
axes[0].invert_yaxis()
axes[0].set_ylabel("Safe categories (top) / unsafe contrasts (bottom)", fontsize=8)
fig.legend(handles=[Patch(color=c,label=l.replace("_"," ")) for l,c in zip(labels,colors)], loc="lower center", ncol=3, fontsize=7.5, frameon=False)
fig.supxlabel("Fraction of raw AI judge labels (25 prompts per category per policy)", fontsize=8, y=.105)
fig.subplots_adjust(left=.29,right=.99,top=.93,bottom=.19,wspace=.14)
output = ROOT / "figures/task4/category_by_policy.png"
output.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(output,dpi=300)
# Keep the filename used by main.tex and the nested legacy path in sync.
for alias in [ROOT / "figures/task4_category_breakdown.png", ROOT / "figures/task4/category_breakdown.png"]:
    alias.write_bytes(output.read_bytes())
plt.close(fig)
print(output)

# Eight separate panels preserve all required trajectories without mixing loss scales.
for task, folder, specs in [
    (2, "task2_ppo", [("learned_reward_mean", "Learned reward"), ("kl_token_mean", "KL / token"), ("policy_loss", "Policy loss"), ("value_loss", "Value loss"), ("entropy_token_mean", "Entropy / token"), ("clip_fraction", "Clip fraction"), ("policy_grad_norm", "Policy grad. norm"), ("response_length_mean", "Length (tokens)")]),
    (3, "task3_grpo", [("reward_mean", "Learned reward"), ("kl_token_mean", "KL / token"), ("policy_loss", "Policy loss"), ("within_group_reward_std_mean", "Reward s.d. / group"), ("diag_sample_entropy", "Sample entropy"), ("uninformative_group_fraction", "Zero-spread groups"), ("gradient_norm", "Gradient norm"), ("response_length_mean", "Length (tokens)")]),
]:
    rows = [json.loads(line) for line in (ROOT.parent / "results" / folder / "standard_updates.jsonl").read_text().splitlines() if line.strip()]
    fig, axes = plt.subplots(2,4,figsize=(6.5,3.1),layout="constrained")
    for ax,(key,title) in zip(axes.flat,specs):
        ax.plot([r["update"] for r in rows],[r[key] for r in rows],color="#0072B2",linewidth=1,marker=".",markersize=3)
        ax.set_title(title,fontsize=8)
        ax.set_xlabel("Update",fontsize=8)
        ax.tick_params(labelsize=8)
        ax.yaxis.get_offset_text().set_fontsize(8)
        ax.grid(alpha=.2)
        ax.spines[["top","right"]].set_visible(False)
    output=ROOT / f"figures/task{task}/{'ppo' if task==2 else 'grpo'}_continuation.png"
    fig.savefig(output,dpi=300)
    plt.close(fig)
    print(output)
