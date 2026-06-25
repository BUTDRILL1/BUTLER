import pyaudio
p = pyaudio.PyAudio()
print('--- Host APIs ---')
for i in range(p.get_host_api_count()):
    print(f"Host API {i}: {p.get_host_api_info_by_index(i).get('name')}")
print('--- Input Devices ---')
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info.get('maxInputChannels') > 0 and 'mic' in info.get('name', '').lower():
        api_name = p.get_host_api_info_by_index(info.get('hostApi'))['name']
        print(f"[{i}] {info.get('name')} (API: {api_name})")
p.terminate()
