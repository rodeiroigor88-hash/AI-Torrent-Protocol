from PIL import Image

def convert_to_ico(input_path, output_path):
    img = Image.open(input_path)
    # The image must be square. The generated one might be 1:1.
    img.save(output_path, format='ICO', sizes=[(256, 256), (128, 128), (64, 64), (32, 32)])
    print(f"Icon saved to {output_path}")

if __name__ == "__main__":
    convert_to_ico(r"C:\Users\Igor2\.gemini\antigravity\brain\c052afda-05b8-4ecc-81c7-364e58af93e2\ghost_network_p2p_1786202834222.jpg", "ghost.ico")
