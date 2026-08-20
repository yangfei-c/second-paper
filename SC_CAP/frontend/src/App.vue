<script setup>
import { computed, ref } from "vue";
import {
  finishSecond,
  firstFeedback,
  firstNext,
  firstRecommend,
  sendSecondFeedback,
  startSecond,
} from "./api";
import FinalSurvey from "./components/FinalSurvey.vue";
import FirstInputPanel from "./components/FirstInputPanel.vue";
import FirstResult from "./components/FirstResult.vue";
import SecondInputPanel from "./components/SecondInputPanel.vue";
import SecondResult from "./components/SecondResult.vue";
import { usePlayback } from "./composables/usePlayback";
import { strategies } from "./constants";

const activePaper = ref("first");
const loading = ref(false);
const feedbackLoading = ref(false);
const error = ref("");
const message = ref("");
const playback = usePlayback();

const firstText = ref("我今天有点累，想听点安慰我的音乐");
const firstStrategy = ref("comfort");
const strength = ref(0.5);
const firstResult = ref(null);
const firstPreference = ref(3);
const firstEffect = ref(3);

const secondText = ref("今天工作了一整天，特别累，一点精神都没有。");
const secondStrategy = ref("energize");
const userInitialV = ref(0);
const userInitialA = ref(0);
const secondSession = ref(null);
const feltV = ref(0);
const feltA = ref(0);
const strategyRating = ref(3);
const secondPreference = ref(3);

const showSurvey = ref(false);
const finalLoading = ref(false);
const finalRatings = ref({
  overall_strategy_fit: 3,
  satisfaction: 3,
  enjoyment: 3,
  smoothness: 3,
  willingness_to_use_again: 3,
});

const hasActiveResult = computed(() => (
  activePaper.value === "first" ? Boolean(firstResult.value) : Boolean(secondSession.value)
));
const loadingText = computed(() => (
  activePaper.value === "first" ? "正在分析情绪并检索单曲…" : "正在初始化 SC-CAP 轨迹…"
));

function strategyLabel(value) {
  return strategies.find((item) => item.value === value)?.label || value;
}

function clearStatus() {
  error.value = "";
  message.value = "";
}

function resetInteraction() {
  clearStatus();
  playback.reset();
}

function selectPaper(paper) {
  if (loading.value || feedbackLoading.value) return;
  activePaper.value = paper;
  resetInteraction();
}

function normalizeSecond(data) {
  return {
    sessionId: data.session_id,
    strategyLabel: strategyLabel(data.strategy),
    textPredVa: data.text_pred_va,
    planned: [data.text_pred_va, data.recommendation.track.pred_va],
    actual: [data.user_initial_va],
    recommendation: data.recommendation,
    strategyQuestion: data.strategy_question,
    completedSteps: 0,
  };
}

async function submitFirst() {
  loading.value = true;
  firstResult.value = null;
  resetInteraction();
  try {
    firstResult.value = await firstRecommend({
      text: firstText.value,
      strategy: firstStrategy.value,
      strength: strength.value,
    });
  } catch (requestError) {
    error.value = requestError.message;
  } finally {
    loading.value = false;
  }
}

async function nextFirst() {
  if (!firstResult.value) return;
  loading.value = true;
  clearStatus();
  try {
    firstResult.value = await firstNext(firstResult.value.session_id, {
      play_duration_seconds: playback.duration.value,
      repeat_count: playback.repeatCount.value,
    });
    playback.reset();
  } catch (requestError) {
    error.value = requestError.message;
  } finally {
    loading.value = false;
  }
}

async function submitFirstFeedback() {
  if (!firstResult.value) return;
  feedbackLoading.value = true;
  clearStatus();
  try {
    await firstFeedback({
      session_id: firstResult.value.session_id,
      song_id: firstResult.value.track.song_id,
      music_preference: firstPreference.value,
      regulation_effect: firstEffect.value,
      play_duration_seconds: playback.duration.value,
      repeat_count: playback.repeatCount.value,
    });
    message.value = "反馈已记录。";
  } catch (requestError) {
    error.value = requestError.message;
  } finally {
    feedbackLoading.value = false;
  }
}

async function submitSecond() {
  loading.value = true;
  secondSession.value = null;
  resetInteraction();
  try {
    const data = await startSecond({
      text: secondText.value,
      strategy: secondStrategy.value,
      user_initial_v: userInitialV.value,
      user_initial_a: userInitialA.value,
    });
    feltV.value = userInitialV.value;
    feltA.value = userInitialA.value;
    secondSession.value = normalizeSecond(data);
  } catch (requestError) {
    error.value = requestError.message;
  } finally {
    loading.value = false;
  }
}

async function submitFeltFeedback() {
  if (!secondSession.value) return;
  feedbackLoading.value = true;
  clearStatus();
  try {
    const data = await sendSecondFeedback(secondSession.value.sessionId, {
      user_felt_v: feltV.value,
      user_felt_a: feltA.value,
      strategy_rating: strategyRating.value,
      music_preference: secondPreference.value,
      play_duration_seconds: playback.duration.value,
      repeat_count: playback.repeatCount.value,
    });
    secondSession.value.actual = data.actual_user_trajectory;
    secondSession.value.planned = data.planned_music_trajectory;
    secondSession.value.completedSteps += 1;
    secondSession.value.recommendation = data.recommendation;
    secondSession.value.strategyQuestion = data.strategy_question;
    feltV.value = data.actual_user_trajectory.at(-1)[0];
    feltA.value = data.actual_user_trajectory.at(-1)[1];
    message.value = data.complete
      ? "第 4 步感受已记录。"
      : "真实感受已记录，已根据当前状态推荐下一首。";
    playback.reset();
  } catch (requestError) {
    error.value = requestError.message;
  } finally {
    feedbackLoading.value = false;
  }
}

async function saveFinal() {
  if (!secondSession.value) return;
  finalLoading.value = true;
  clearStatus();
  try {
    await finishSecond(secondSession.value.sessionId, finalRatings.value);
    showSurvey.value = false;
    message.value = "本次用户实验记录已保存。";
  } catch (requestError) {
    error.value = requestError.message;
  } finally {
    finalLoading.value = false;
  }
}
</script>

<template>
  <main class="page">
    <header class="app-header">
      <h1><span class="title-icon">♪</span> 情绪调节型音乐推荐</h1>
      <div class="paper-tabs">
        <button
          :class="{ active: activePaper === 'first' }"
          :disabled="loading || feedbackLoading"
          @click="selectPaper('first')"
        >
          第一篇
        </button>
        <button
          :class="{ active: activePaper === 'second' }"
          :disabled="loading || feedbackLoading"
          @click="selectPaper('second')"
        >
          第二篇
        </button>
      </div>
    </header>

    <div class="workspace">
      <FirstInputPanel
        v-if="activePaper === 'first'"
        v-model:text="firstText"
        v-model:strategy="firstStrategy"
        v-model:strength="strength"
        :loading="loading"
        :error="error"
        @submit="submitFirst"
      />
      <SecondInputPanel
        v-else
        v-model:text="secondText"
        v-model:strategy="secondStrategy"
        v-model:user-initial-v="userInitialV"
        v-model:user-initial-a="userInitialA"
        :loading="loading"
        :error="error"
        @submit="submitSecond"
      />

      <section class="panel result-panel">
        <div class="panel-title">
          <h2><span class="icon">♫</span> 推荐音乐</h2>
          <span class="session-status">
            {{ activePaper === "first" ? "单曲推荐" : "连续反馈调节" }}
          </span>
        </div>

        <FirstResult
          v-if="activePaper === 'first' && firstResult"
          v-model:music-preference="firstPreference"
          v-model:regulation-effect="firstEffect"
          :result="firstResult"
          :duration="playback.duration"
          :repeat-count="playback.repeatCount"
          :next-loading="loading"
          :feedback-loading="feedbackLoading"
          :message="message"
          @next="nextFirst"
          @feedback="submitFirstFeedback"
          @play="playback.play"
          @timeupdate="playback.timeupdate"
          @ended="playback.finish"
        />
        <SecondResult
          v-else-if="activePaper === 'second' && secondSession"
          v-model:felt-v="feltV"
          v-model:felt-a="feltA"
          v-model:strategy-rating="strategyRating"
          v-model:music-preference="secondPreference"
          :session="secondSession"
          :duration="playback.duration"
          :repeat-count="playback.repeatCount"
          :audio-ended="playback.ended"
          :feedback-loading="feedbackLoading"
          :complete="!secondSession.recommendation"
          :message="message"
          @feedback="submitFeltFeedback"
          @finish="showSurvey = true"
          @play="playback.play"
          @timeupdate="playback.timeupdate"
          @ended="playback.finish"
        />
        <div v-else-if="loading" class="empty-state request-state" aria-live="polite">
          <div class="state-copy">
            <span class="state-spinner" aria-hidden="true"></span>
            <h3>{{ loadingText }}</h3>
            <p>首次加载冻结模型可能需要几秒钟，请稍候。</p>
          </div>
        </div>
        <div v-else-if="error" class="empty-state error-state" role="alert">
          <div class="state-copy">
            <h3>推荐请求未完成</h3>
            <p>{{ error }}</p>
          </div>
        </div>
        <div v-else class="empty-state">
          <h3><span class="icon">♫</span> 等待推荐</h3>
        </div>

        <p v-if="error && hasActiveResult" class="error result-error" role="alert">
          {{ error }}
        </p>
      </section>
    </div>
  </main>

  <FinalSurvey
    v-if="showSurvey"
    v-model:ratings="finalRatings"
    :loading="finalLoading"
    @submit="saveFinal"
  />
</template>
