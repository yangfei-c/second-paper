<script setup>
import { computed } from "vue";
import VaTrajectoryPlot from "./VaTrajectoryPlot.vue";
import { formatVa } from "../utils/vaPlot";

const musicPreference = defineModel("musicPreference");
const regulationEffect = defineModel("regulationEffect");
const props = defineProps({ result: Object, duration: Number, repeatCount: Number, nextLoading: Boolean, feedbackLoading: Boolean, message: String });
defineEmits(["next", "feedback", "play", "timeupdate", "ended"]);
const planned = computed(() => props.result ? [props.result.initial_va, props.result.track.va] : []);
</script>

<template>
  <div v-if="result" class="result-content first-result-content">
    <div class="va-summary"><div><span>初始 VA</span><strong>{{ formatVa(result.initial_va) }}</strong></div><div><span>目标 VA</span><strong>{{ formatVa(result.target_va) }}</strong></div><div><span>音乐 VA</span><strong>{{ formatVa(result.track.va) }}</strong></div></div>
    <div class="result-main"><VaTrajectoryPlot :planned="planned" :actual="[]" />
      <article class="track"><div class="track-head"><div><span class="eyebrow">当前歌曲</span><h3>{{ result.track.song_id }}</h3></div><button class="secondary-button swap-button" @click="$emit('next')" :disabled="nextLoading || !result.has_next"><span class="button-icon">↻</span>{{ nextLoading ? "换歌中" : result.has_next ? "换一首" : "无候选" }}</button></div>
      <p class="explanation">{{ result.explanation }}</p><p class="tags">{{ result.track.display_tags?.join(" · ") }}</p>
      <audio v-if="result.track.audio_url" :key="result.track.song_id" :src="result.track.audio_url" controls @play="$emit('play', $event)" @timeupdate="$emit('timeupdate', $event)" @ended="$emit('ended', $event)"></audio><p v-else class="error">这首歌缺少可播放地址。</p></article></div>
    <form class="feedback" @submit.prevent="$emit('feedback')"><div class="feedback-head"><h3><span class="icon">✓</span> 反馈</h3><p class="playback">已播放 {{ duration.toFixed(1) }} 秒 · 重复播放 {{ repeatCount }} 次</p></div><div class="feedback-grid"><label>音乐偏好<select v-model="musicPreference"><option :value="1">1 很不喜欢</option><option :value="2">2 不太喜欢</option><option :value="3">3 一般</option><option :value="4">4 喜欢</option><option :value="5">5 很喜欢</option></select></label><label>调节效果<select v-model="regulationEffect"><option :value="1">1 没有效果</option><option :value="2">2 效果较弱</option><option :value="3">3 一般</option><option :value="4">4 有帮助</option><option :value="5">5 很有帮助</option></select></label></div><button class="primary-button" :disabled="feedbackLoading"><span class="button-icon">✓</span>{{ feedbackLoading ? "提交中..." : "提交反馈" }}</button><p v-if="message" class="success">{{ message }}</p></form>
  </div>
</template>
