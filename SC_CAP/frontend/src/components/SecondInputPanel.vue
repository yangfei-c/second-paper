<script setup>
import { strategies } from "../constants";
import VaInput from "./VaInput.vue";

const text = defineModel("text");
const strategy = defineModel("strategy");
const userInitialV = defineModel("userInitialV");
const userInitialA = defineModel("userInitialA");
defineProps({ loading: Boolean, error: String });
defineEmits(["submit"]);
</script>

<template>
  <section class="panel input-panel second-input-panel">
    <div class="panel-title"><h2><span class="icon">✎</span> 输入</h2><span class="session-status">4 首连续调节</span></div>
    <label>情绪文本</label>
    <textarea v-model="text" rows="6" placeholder="例如：今天工作了一整天，特别累，一点精神都没有。"></textarea>
    <label>调节策略</label>
    <select v-model="strategy"><option v-for="item in strategies" :key="item.value" :value="item.value">{{ item.label }}</option></select>
    <div class="notice"><strong>记录真实初始感受</strong><span>仅用于记录你的真实轨迹，不参与第一首音乐推荐。第一首只由文本 VA 预测决定。</span></div>
    <VaInput v-model:value-v="userInitialV" v-model:value-a="userInitialA" compact />
    <button class="primary-button" @click="$emit('submit')" :disabled="loading"><span class="button-icon">▶</span>{{ loading ? "正在分析与推荐..." : "开始连续推荐" }}</button>
    <p v-if="error" class="error">{{ error }}</p>
  </section>
</template>
