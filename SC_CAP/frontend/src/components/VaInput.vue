<script setup>
const valueV = defineModel("valueV");
const valueA = defineModel("valueA");
defineProps({ compact: Boolean });
</script>

<template>
  <div class="va-input" :class="{ compact }">
    <div class="va-help">
      <strong>V 效价</strong><span>-1 不愉快 · 0 中性 · +1 愉快</span>
      <strong>A 唤醒</strong><span>-1 平静/低能量 · 0 中等 · +1 兴奋/高能量</span>
    </div>
    <label>此刻的 V：{{ Number(valueV).toFixed(2) }}<input v-model.number="valueV" type="range" min="-1" max="1" step="0.05" /></label>
    <label>此刻的 A：{{ Number(valueA).toFixed(2) }}<input v-model.number="valueA" type="range" min="-1" max="1" step="0.05" /></label>
  </div>
</template>
