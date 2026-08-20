<script setup>
import { computed } from "vue";
import { strategies } from "../constants";

const text = defineModel("text");
const strategy = defineModel("strategy");
const strength = defineModel("strength");
defineProps({ loading: Boolean, error: String });
defineEmits(["submit"]);
const strengthText = computed(() => Number(strength.value) < .34 ? "弱" : Number(strength.value) < .67 ? "中" : "强");
</script>

<template>
  <section class="panel input-panel">
    <div class="panel-title"><h2><span class="icon">✎</span> 输入</h2></div>
    <label>情绪文本</label>
    <textarea v-model="text" rows="7" placeholder="例如：我今天有点累，想听点安慰我的音乐"></textarea>
    <label>调节策略</label>
    <select v-model="strategy"><option v-for="item in strategies" :key="item.value" :value="item.value">{{ item.label }}</option></select>
    <div class="strength-block">
      <label>调节强度：{{ Number(strength).toFixed(2) }} · {{ strengthText }}</label>
      <input v-model.number="strength" type="range" min="0" max="1" step="0.05" />
      <div class="strength-labels"><span>弱</span><span>中</span><span>强</span></div>
    </div>
    <button class="primary-button" @click="$emit('submit')" :disabled="loading"><span class="button-icon">▶</span>{{ loading ? "正在推荐..." : "开始推荐" }}</button>
    <p v-if="error" class="error">{{ error }}</p>
  </section>
</template>
