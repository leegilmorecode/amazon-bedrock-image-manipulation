import { BedrockImageManipulation, type TaskType } from '../src/index';

import fs from 'node:fs';
import path from 'node:path';

const readImageAsBase64 = (filename: string): string => {
  const imagePath = path.join(__dirname, '..', 'images', filename);
  const imageBuffer = fs.readFileSync(imagePath);
  return imageBuffer.toString('base64');
};

const writeImageFromBase64 = (base64Data: string, outputFilename: string) => {
  const outputPath = path.join(__dirname, '..', 'results', outputFilename);
  const buffer = Buffer.from(base64Data, 'base64');
  fs.writeFileSync(outputPath, buffer);
};

const run = async () => {
  const imageManipulator = new BedrockImageManipulation();

  const tasks: {
    taskType: TaskType;
    inputFiles: string[];
    outputFile: string;
    prompt?: string;
    configOverrides?: Partial<
      Parameters<typeof imageManipulator.generateImage>[0]
    >;
  }[] = [
    {
      // this will generate an image from a text prompt of a futuristic cityscape
      taskType: 'TEXT_IMAGE',
      inputFiles: [],
      outputFile: 'result-1.png',
      prompt: 'A beautiful sunrise over a futuristic cityscape',
    },
    {
      // this will replace the area of the image (tree) with a palm tree
      taskType: 'INPAINTING',
      inputFiles: ['file-2.png'], // a second image would be the mask image i.e. 'file-2-mask.png'
      outputFile: 'result-2.png',
      prompt: 'A palm tree', // note this will replace roughly the same area as the original element i.e. the tree
      configOverrides: {
        maskPrompt: 'Tree', // you can only use maskPrompt or maskImage, not both
        negativePrompt: 'forest',
        imageConfig: {
          width: 1024,
          height: 1024,
        },
      },
    },
    {
      // this will replace the area of the image (coffee cup) with a bowl of olives
      taskType: 'INPAINTING',
      inputFiles: ['file-3.png', 'file-3-mask.png'], // a second image would be the mask image i.e. 'file-3-mask.png'
      outputFile: 'result-3.png',
      prompt: 'A bowl of olives', // note this will replace the mask area
      configOverrides: {
        maskPrompt: undefined, // you can only use maskPrompt or maskImage, not both
        negativePrompt: 'coffee',
        imageConfig: {
          width: 1024,
          height: 1024,
        },
      },
    },
    {
      // this will replace the background of the image from the countryside to a cityscape but leave the man ni the middle
      taskType: 'OUTPAINTING',
      inputFiles: ['file-4.png'],
      outputFile: 'result-4.png',
      prompt:
        'A man walking in the middle of a busy New York street with skyscrapers in the background and people around him',
      configOverrides: {
        maskPrompt: 'man', // this will replace the area around
        imageConfig: {
          width: 1024,
          height: 1024,
        },
      },
    },
    {
      taskType: 'BACKGROUND_REMOVAL',
      inputFiles: ['file-5.png'],
      outputFile: 'result-5.png',
    },
    {
      taskType: 'COLOR_GUIDED_GENERATION',
      inputFiles: ['file-6.png'],
      outputFile: 'result-6.png',
      prompt:
        'A small blonde child smiling at the camera showing their head and shoulders, where they are looking left',
      configOverrides: {
        colorHexList: ['#5e17eb'], // purple
        controlStrength: 1,
        imageConfig: {
          seed: 100,
        },
      },
    },
  ];

  for (const [index, task] of tasks.entries()) {
    console.log(`Running task ${index + 1}: ${task.taskType}`);

    const inputImage = task.inputFiles[0]
      ? readImageAsBase64(task.inputFiles[0])
      : undefined;

    const maskImage =
      (task.taskType === 'INPAINTING' || task.taskType === 'OUTPAINTING') &&
      !task.configOverrides?.maskPrompt
        ? readImageAsBase64(task.inputFiles[1])
        : undefined;

    const generationResult = await imageManipulator.generateImage({
      modelId: 'amazon.nova-canvas-v1:0',
      taskType: task.taskType,
      prompt: task.prompt,
      base64Image: inputImage,
      maskImage: maskImage,
      imageConfig: {
        width: 512,
        height: 512,
        quality: 'premium',
      },
      ...task.configOverrides,
    });

    writeImageFromBase64(generationResult.base64Image, task.outputFile);
  }

  console.log('All tasks complete. Results saved to /results folder.');
};

run().catch((error) => {
  console.error('Failed to generate image:', error);
});
