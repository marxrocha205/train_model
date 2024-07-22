import os
from trainer import Trainer, TrainerArgs
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

def main():
    # Caminhos dos arquivos
    output_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(output_path, "wavs")

    # Configuração do dataset
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",
        meta_file_train="metadata.csv",
        path=dataset_path
    )

    # Configuração de áudio
    audio_config = BaseAudioConfig(
        sample_rate=22050,
        resample=True,
        do_trim_silence=True,
        trim_db=30.0  # Ajustar valor de trim_db para acomodar variações
    )

    # Configuração do modelo
    config = GlowTTSConfig(
        batch_size=16,  # Reduzir o batch size para ajudar na carga de memória
        eval_batch_size=8,
        num_loader_workers=2,
        num_eval_loader_workers=2,
        precompute_num_workers=2,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="pt-br",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        use_speaker_embedding=True,
        min_text_len=1,  # Ajuste conforme necessário
        max_text_len=100,  # Ajuste conforme necessário
        min_audio_len=1000,  # Ajuste conforme necessário
        max_audio_len=300000,  # Ajuste conforme necessário
        eval_split_size=0.1,
    )

    # Inicializa o processador de áudio
    ap = AudioProcessor.init_from_config(config)

    # Inicializa o tokenizador
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Carrega as amostras de dados
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # Verificações para garantir que as amostras foram carregadas corretamente
    if train_samples is None:
        print("Erro: train_samples é None.")
    else:
        print(f"train_samples carregados com sucesso. Número de amostras: {len(train_samples)}")

    if eval_samples is None:
        print("Erro: eval_samples é None.")
    else:
        print(f"eval_samples carregados com sucesso. Número de amostras: {len(eval_samples)}")

    # Inicializa o gerenciador de locutores
    speaker_manager = SpeakerManager()
    if train_samples is not None and eval_samples is not None:
        speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
        config.num_speakers = speaker_manager.num_speakers

        # Inicializa o modelo
        model = GlowTTS(config, ap, tokenizer, speaker_manager=speaker_manager)

        # Inicializa o treinador
        trainer = Trainer(
            TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
        )

        # Inicia o treinamento
        trainer.fit()
    else:
        print("Erro ao carregar as amostras de treinamento ou avaliação.")

if __name__ == "__main__":
    main()
