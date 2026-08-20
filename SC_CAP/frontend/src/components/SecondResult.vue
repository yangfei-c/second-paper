<script setup>
import { computed } from "vue";
import VaInput from "./VaInput.vue";
import VaTrajectoryPlot from "./VaTrajectoryPlot.vue";
import { formatVa } from "../utils/vaPlot";

const feltV = defineModel("feltV");
const feltA = defineModel("feltA");
const strategyRating = defineModel("strategyRating");
const musicPreference = defineModel("musicPreference");
const props = defineProps({ session: Object, duration: Number, repeatCount: Number, audioEnded: Boolean, feedbackLoading: Boolean, complete: Boolean, message: String });
defineEmits(["feedback", "finish", "play", "timeupdate", "ended"]);
const recommendation = computed(() => props.session?.recommendation);
const track = computed(() => recommendation.value?.track);
const currentStep = computed(() => recommendation.value?.step || props.session?.completedSteps || 0);
const strategyQuestion = computed(() => props.session?.strategyQuestion || "");
</script>

<template>
  <div v-if="session" class="result-content second-result-content">
    <div class="va-summary"><div><span>文本 VA（首曲依据）</span><strong>{{ formatVa(session.textPredVa) }}</strong></div><div><span>当前音乐 VA</span><strong>{{ track ? formatVa(track.pred_va) : "已完成" }}</strong></div><div><span>步骤</span><strong>{{ complete ? "4 / 4 完成" : `${currentStep} / 4` }}</strong></div></div>
    <div class="result-main"><VaTrajectoryPlot :planned="session.planned" :actual="session.actual" />
      <article class="track"><div class="track-head"><div><span class="eyebrow">{{ complete ? "连续调节已完成" : `当前歌曲 ${currentStep} / 4` }}</span><h3>{{ track?.song_id || "完成本次序列" }}</h3></div><span class="session-status">{{ session.strategyLabel }}</span></div>
      <template v-if="track"><p class="explanation">这首歌满足当前策略和累计轨迹约束。听完后请报告你自己此刻的真实感受，系统会据此推荐下一首。</p><p class="tags">{{ track.display_tags?.join(" · ") || "Catalog 音乐" }}</p><audio v-if="track.audio_url" :key="track.song_id" :src="track.audio_url" controls @play="$emit('play', $event)" @timeupdate="$emit('timeupdate', $event)" @ended="$emit('ended', $event)"></audio><p v-else class="error">这首歌缺少可播放地址。</p></template><template v-else><p class="explanation">四首音乐均已完成。请提交总体评价以保存本次用户实验记录。</p><button class="primary-button" @click="$emit('finish')">提交总体评价</button></template></article></div>
    <div v-if="!complete && !audioEnded" class="await-feedback">请完整播放当前音乐。播放结束后将显示真实感受记录。</div>
    <form v-else-if="!complete" class="feedback felt-feedback" @submit.prevent="$emit('feedback')"><div class="feedback-head"><h3><span class="icon">✓</span> 听后真实感受</h3><p class="playback">已播放 {{ duration.toFixed(1) }} 秒 · 重复播放 {{ repeatCount }} 次</p></div><p class="felt-instruction">相较于听歌前，你自己的 V 和 A 有怎样的变化？请标记你此刻真实的感受，而不是评价这首音乐听起来是什么情绪。</p><VaInput v-model:value-v="feltV" v-model:value-a="feltA" /><div class="feedback-grid"><label>{{ strategyQuestion }}<select v-model="strategyRating"><option :value="1">1 完全没有</option><option :value="2">2 较弱</option><option :value="3">3 一般</option><option :value="4">4 明显</option><option :value="5">5 非常明显</option></select></label><label>这首音乐你喜欢吗？<select v-model="musicPreference"><option :value="1">1 很不喜欢</option><option :value="2">2 不太喜欢</option><option :value="3">3 一般</option><option :value="4">4 喜欢</option><option :value="5">5 很喜欢</option></select></label></div><button class="primary-button" :disabled="feedbackLoading"><span class="button-icon">✓</span>{{ feedbackLoading ? "正在重新规划..." : "提交感受并推荐下一首" }}</button><p v-if="message" class="success">{{ message }}</p></form>
  </div>
</template>
