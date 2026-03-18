import os
import glob
from pydub import AudioSegment

# Папки, которые нужно обработать (дроны пропускаем)
folders_to_process = ['background_train']
chunk_length_ms = 1000  # 1 секунда в миллисекундах

def process_audio_files():
    for folder in folders_to_process:
        if not os.path.exists(folder):
            print(f"Папка {folder} не найдена, пропускаем.")
            continue
            
        output_folder = f"{folder}_1sec"
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"\n=== Обработка папки: {folder} -> {output_folder} ===")
        
        # Ищем аудиофайлы (wav, mp3, flac, ogg)
        files = []
        for ext in ('*.wav', '*.mp3', '*.flac', '*.ogg'):
            files.extend(glob.glob(os.path.join(folder, ext)))
            
        for filepath in files:
            filename = os.path.basename(filepath)
            
            # Пропускаем файлы, которые мы уже нарезали (защита от повторного запуска)
            if "_chunk_" in filename:
                continue
                
            try:
                audio = AudioSegment.from_file(filepath)
                duration_ms = len(audio)
                
                # Если файл уже длится около 1 секунды (с небольшой погрешностью), пропускаем
                if duration_ms <= chunk_length_ms + 100:
                    continue
                    
                print(f"Нарезка: {filename} (Длительность: {duration_ms/1000:.1f}с)")
                
                chunk_count = 0
                for i in range(0, duration_ms, chunk_length_ms):
                    chunk = audio[i:i + chunk_length_ms]
                    
                    # Обработка последнего кусочка, если он меньше 1 секунды
                    if len(chunk) < chunk_length_ms:
                        # Если остаток меньше 0.5 сек - просто выбрасываем его как неинформативный
                        if len(chunk) < 500:
                            continue
                        # Если больше 0.5 сек - добиваем тишиной до ровно 1 секунды (важно для одинакового shape в нейросети)
                        else:
                            silence = AudioSegment.silent(duration=chunk_length_ms - len(chunk))
                            chunk = chunk + silence
                            
                    # Сохраняем кусок
                    chunk_name = f"{os.path.splitext(filename)[0]}_chunk_{chunk_count}.wav"
                    chunk_path = os.path.join(output_folder, chunk_name)
                    chunk.export(chunk_path, format="wav")
                    chunk_count += 1
                    
            except Exception as e:
                print(f"Ошибка при обработке {filepath}: {e}")

if __name__ == "__main__":
    process_audio_files()
    print("\nГотово! Все длинные аудиофайлы разбиты на секундные отрезки.")
