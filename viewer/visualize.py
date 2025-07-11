import tkinter as tk
from PIL import Image, ImageTk
import torchvision
from datasets.cifar10 import get_dataloaders

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class ImageViewer:
    def __init__(self, root, dataloader):
        self.root = root
        self.dataloader = dataloader
        self.images, self.labels = self.load_all_images()
        self.page = 0
        self.page_size = 16

        self.frame = tk.Frame(root)
        self.frame.pack() # 放置主框架

        self.nav = tk.Frame(root)
        self.nav.pack()

        self.prev_button = tk.Button(self.nav, text="上一页", command=self.prev_page)
        self.prev_button.pack(side=tk.LEFT, padx=10)

        self.next_button = tk.Button(self.nav, text="下一页", command=self.next_page)
        self.next_button.pack(side=tk.LEFT, padx=10)

        self.show_page()

    def load_all_images(self): # 只从一个batch中加载图片和标签
        images = []
        labels = []
        for imgs, lbls in self.dataloader:
            for i in range(imgs.size(0)):
                unnormalize = torchvision.transforms.Normalize( # 反归一化
                    mean=[-m / s for m, s in zip((0.4914, 0.4822, 0.4465),
                                                 (0.2023, 0.1994, 0.2010))],
                    std=[1 / s for s in (0.2023, 0.1994, 0.2010)]
                )
                img = torchvision.transforms.ToPILImage()(unnormalize(imgs[i]))
                images.append(img)
                labels.append(lbls[i].item())
            break
        return images, labels

    def clear_frame(self): # 清除当前页面的图像和标签
        for widget in self.frame.winfo_children():
            widget.destroy()

    def show_page(self):
        self.clear_frame()
        start = self.page * self.page_size
        end = start + self.page_size # 根据当前页码，确定要显示的图像范围

        for i, img in enumerate(self.images[start:end]):
            row = i // 4
            col = i % 4
            resized = img.resize((64, 64))
            photo = ImageTk.PhotoImage(resized)

            panel = tk.Label(self.frame, image=photo)
            panel.image = photo
            panel.grid(row=row * 2, column=col)

            label_text = f"真实: {CIFAR10_CLASSES[self.labels[start + i]]}"
            label = tk.Label(self.frame, text=label_text)
            label.grid(row=row * 2 + 1, column=col)

    def next_page(self):
        if (self.page + 1) * self.page_size < len(self.images):
            self.page += 1
            self.show_page()

    def prev_page(self):
        if self.page > 0:
            self.page -= 1
            self.show_page()


if __name__ == '__main__':
    train_loader, _, _, _ = get_dataloaders()

    root = tk.Tk()
    root.title("CIFAR-10 图像查看器")
    app = ImageViewer(root, train_loader)
    root.mainloop()
