from visualizers.Visualizer import Visualizer


if __name__ == "__main__":
    visualizer = Visualizer("kdef", "../dataset_kdef", "jpg")

    # red: pose1 landmarks
    pose1 = "1011-neutral"
    # blue: pose2 landmarks
    pose2 = "1011-afraid"

    visualizer.compare(pose1, pose2)
