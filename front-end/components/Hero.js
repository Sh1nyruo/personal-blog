import Link from 'next/link'
import { useState } from 'react'
import { IoLogoGithub, IoLogoYoutube, IoMail, IoCall } from 'react-icons/io5'
import { FaBaby } from 'react-icons/fa';
import Notification from './Notification'

function SocialLink({ icon: Icon, ...props }) {
  return (
    <Link className="-m-1 p-1 " {...props}>
      <Icon className="h-6 w-6 cursor-pointer fill-gray-500 transition hover:fill-gray-200" />
    </Link>
  )
}

function CopyToClipboard({ icon: Icon, text, ...props }) {
  const [show, setShow] = useState(false)

  const handleClick = () => {
    navigator.clipboard.writeText(text.contact)
    setShow(!show)

    setTimeout(() => {
      setShow(false)
    }, 3000)
  }

  return (
    <div className="-m-1 p-1 " {...props}>
      <Icon
        className="h-6 w-6 cursor-pointer fill-gray-500 transition hover:fill-gray-200"
        onClick={handleClick}
      />
      <Notification show={show} setShow={setShow} text={text} />
    </div>
  )
}

export default function Hero() {
  return (
    <div className="mb-5 max-w-2xl">
      <h1 className="text-4xl font-bold tracking-tight text-gray-800 dark:text-zinc-100 sm:text-5xl">
        Web开发者，技术爱好者，科学麻将爱好者
      </h1>
      <p className="mt-6 text-base text-gray-600 dark:text-gray-400">
        我是顾永威，现在正在上海财经大学计算机科学与技术专业就读。作为一名入门的Web开发者，我正在努力学习
        Next.js, Node.js和TypeScript。我对前端开发和后端开发都很感兴趣，希望能够成为一名全栈开发者。
      </p>
      <div className="mt-6 flex gap-6">
        <SocialLink
          href="https://github.com/Sh1nyruo"
          aria-label="我的GitHub"
          icon={IoLogoGithub}
        />
        <SocialLink
          href="https://space.bilibili.com/7807638?spm_id_from=333.337.0.0"
          aria-label="我的Bilibili"
          icon={IoLogoYoutube}
        />
        <CopyToClipboard
          text={{ contact: 'guyongwei@163.sufe.edu.cn', type: 'Email' }}
          aria-label="给我发送emial"
          icon={IoMail}
        />
        <CopyToClipboard
          text={{ contact: '+86 17701702984', type: 'Phone number' }}
          aria-label="给我打电话"
          icon={IoCall}
        />
      </div>
    </div>
  )
}
// <div className="flex flex-col w-full">
//   <div className="pb-4 space-y-2 text-center md:space-y-5 md:text-left">
//     <PageTitle>Web Developer, Tech Enthusiast, and Fitness Junkie</PageTitle>
//     <p className="pb-4 text-lg leading-7 prose text-gray-400 max-w-none">
//       Technology enthusiast experienced in consumer electronics industry. I believe the optimal
//       code is achieved when the user and development experience is frictionless and intuitive.{' '}
//       <Link href={`mailto:${siteMetadata.email}`}>
//         <a
//           className="font-medium leading-6 "
//           aria-label={`Email to ${siteMetadata.email}`}
//           title={`Email to ${siteMetadata.email}`}
//         >
//           Get in touch &rarr;
//         </a>
//       </Link>
//     </p>
//   </div>
// </div>
